import argparse
import os
import warnings

from transformers import AutoTokenizer

from flexgen.utils import (
    str2bool,
    ExecutionEnv,
    Policy,
    GB, DUMMY_WEIGHT,
    project_decode_latency, write_benchmark_log
)
from flexgen.torch_backend import (TorchDevice,
                                   TorchDisk,
                                   TorchMixedDevice,
                                   AsyncIOManager,
                                   fix_recursive_import)

from flexgen.opt_config import get_opt_config
from flexgen.compression import CompressionConfig
from flexgen.model.model import OptLM
from flexgen.timer import timers
import flexgen.torch_numa as torch_numa

fix_recursive_import()

NUM_NUMAS = torch_numa.get_numa_nodes()
HF_MIRROR = 'https://hf-mirror.com'


def setup_hf_mirror(mirror_url=HF_MIRROR):
    os.environ['HF_ENDPOINT'] = mirror_url

def parse_device_config(device_str):
    """
    Parse device configuration string like 'numa0:50,numa1:30,disk:20'
    Returns dict: {'numa0': 50, 'numa1': 30, 'disk': 20}
    """
    if not device_str:
        return {}
    
    config = {}
    for item in device_str.split(','):
        if ':' not in item:
            raise ValueError(f"Invalid device config format: {item}. Expected 'device:percentage'")
        device, percent_str = item.split(':', 1)
        device = device.strip()
        
        assert device in ['numa'+str(i) for i in range(NUM_NUMAS)] + ['disk'], \
            f"Invalid device: {device}. Expected numa0, numa1, ..., disk"
            
        try:
            percent = int(percent_str.strip())
        except ValueError:
            raise ValueError(f"Invalid percentage: {percent_str}")
        
        if percent < 0 or percent > 100:
            raise ValueError(f"Percentage must be 0-100, got {percent}")
        
        config[device] = percent
    
    # Validate that percentages sum to 100
    total = sum(config.values())
    if total != 100:
        raise ValueError(f"Device percentages must sum to 100, got {total}")
    
    return config


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def get_filename(args):
    """Generate filename for benchmark log"""
    model_size = args.model.split('/')[-1]  # Extract model size from path like "facebook/opt-125m"
    return f"fo-{model_size}-gbs{args.batch_size}-ngbs{args.num_batches}-prompt{args.prompt_len}-gen{args.gen_len}-percent-100-100-100-"


def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    
    # Process device configuration
    device_config = {}
    device_config['weight'] = parse_device_config(args.weight_devices)
    device_config['cache'] = parse_device_config(args.cache_devices)
    device_config['activation'] = parse_device_config(args.activation_devices)
        
    print(f"Device configuration: \n {device_config}")
    
    setup_hf_mirror(args.hf_mirror)
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_batches * args.batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    # Create NUMA-aware execution environment
    env = ExecutionEnv.create_env(args.offload_dir, numa_nodes=list(range(NUM_NUMAS)))
    print(f"Available NUMA nodes: {env.get_available_numa_nodes()}")

    # Create policy using new device configuration
    policy = Policy.create_from_device_config(batch_size=args.batch_size,
                                              num_batches=args.num_batches,
                                              device_config=device_config,
                                              overlap=args.overlap,
                                              sep_layer=args.sep_layer,
                                              attn_sparsity=args.attn_sparsity,
                                              comp_weight=args.compress_weight,
                                              comp_weight_config=CompressionConfig(num_bits=4,
                                                                                   group_size=64,
                                                                                   group_dim=0,
                                                                                   symmetric=False),
                                              comp_cache=args.compress_cache,
                                              comp_cache_config=CompressionConfig(num_bits=4,
                                                                                  group_size=64,
                                                                                  group_dim=2,
                                                                                  symmetric=False))
    
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = OptLM(opt_config, env, args.path, policy)

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        AsyncIOManager().close()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, cpu_peak_mem = env.get_numa_device(env.get_available_numa_nodes()[0]).mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    env.get_numa_device(env.get_available_numa_nodes()[0]).print_stats()
    projected = cut_gen_len

    if args.log_file == "auto":
        # 确保logs目录存在
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        filename = os.path.join(log_dir, get_filename(args) + ".log")
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)





def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload_dir", type=str, default="~/flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--hf_mirror", type=str, default="https://hf-mirror.com",
        help="HuggingFace mirror URL (default: https://hf-mirror.com)")
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--cut_gen_len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=1)
    
    # NUMA-aware device configuration
    parser.add_argument("--weight_devices", type=str, default=None,
        help="Weight placement. Format: 'numa0:50,numa1:30,disk:20'")
    parser.add_argument("--cache_devices", type=str, default=None,
        help="Cache placement. Format: 'numa0:70,numa1:30'")
    parser.add_argument("--activation_devices", type=str, default=None,
        help="Activation placement. Format: 'numa0:100'")
    
    parser.add_argument("--sep_layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--attn_sparsity", type=float, default=1.0)
    parser.add_argument("--compress_weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress_cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log_file", type=str, default="auto")
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)
    


if __name__ == "__main__":
    # 过滤已知的warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    
    assert not args.compress_weight, "compress_weight is Not support Now"

    run_flexllmgen(args)