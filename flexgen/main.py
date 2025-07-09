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
                                   TorchNuma,
                                   AsyncIOManager)

from flexgen.opt_config import get_opt_config
from flexgen.model.model import OptLM
from flexgen.timer import timers

from flexgen.torch_backend import  print_memory_copy_stats


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.batch_size}-" \
               f"ngbs{args.num_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
        
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_batches * args.batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    cpu = TorchDevice("cpu")
    numa = TorchNuma(numa_node=2)
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(cpu=cpu, disk=disk, numa=numa)

    policy = Policy(args.batch_size,
                    args.num_batches,
                    args.percent[0],
                    args.percent[1],
                    args.percent[2],
                    args.percent[3],
                    args.percent[4],
                    args.percent[5],
                    args.overlap,
                    args.sep_layer,
                    args.attn_sparsity)

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
            warmup_inputs, max_new_tokens=1)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len, cut_gen_len=cut_gen_len)
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
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    cpu.print_stats()
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
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--cut_gen_len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 100, 100, 0, 0, 0],
        help="Six integers representing the percentage of "
             "weight_cpu, cache_cpu, act_cpu, "
             "weight_numa, cache_numa, act_numa.")
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
    # 获取当前的 PID 并停顿 10s
    import time
    import os
    pid = os.getpid()
    print(f"Running FlexLLMGen with PID: {pid}")
    time.sleep(10)
    
    
    # 过滤已知的warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    
    assert not args.compress_weight, "compress_weight is Not support Now"

    run_flexllmgen(args)
    
    print_memory_copy_stats()