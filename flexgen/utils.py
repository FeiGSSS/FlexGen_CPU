import dataclasses
from typing import Union, List, Optional, Any
import gc
import argparse

import numpy as np
import torch



KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12
DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    inputs: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int
    cut_gen_len: Optional[int]

    do_sample: bool
    temperature: float
    stop: Optional[int]
    
    
@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    cpu: Any = None
    disk: Any = None
    mixed: Any = None

    @classmethod
    def create(cls, offload_dir):
        # fix recursive import
        from flexgen.torch_backend import TorchDevice, TorchDisk, TorchMixedDevice
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(cpu=cpu, disk=disk, mixed=TorchMixedDevice([cpu, disk]))
    
    
np_dtype_to_torch_dtype = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int32: torch.int32,
    np.int64: torch.int64,
    bool: torch.bool,
}

torch_dtype_to_np_dtype = {
    torch.float16: np.float16, torch.float32: np.float32,
    torch.uint8: np.uint8, torch.int8: np.int8, torch.int32: np.int32,
    torch.int64: np.int64, torch.bool: bool,
}

torch_dtype_to_num_bytes = {
    torch.float16: 2, torch.float32: 4,
    torch.int8: 1, torch.uint8: 1, torch.int32: 4, torch.int64: 8,
    torch.bool: 1,
}


@dataclasses.dataclass(frozen=True)
class Policy:
    batch_size: int
    num_batches: int

    # Legacy percent format for backward compatibility
    w_cpu_percent: float
    cache_cpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Sparsity of attention weights
    attn_sparsity: float
    
    # Compress weights with group-wise quantization
    comp_weight: bool
    comp_weight_config: Any

    # Compress KV cache with group-wise quantization
    comp_cache: bool
    comp_cache_config: Any
    
    # New NUMA-aware device configuration
    device_config: Any = None  # Dict containing device placement configuration

    @property
    def w_disk_percent(self):
        return 100  - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_cpu_percent
    
    def get_device_percent(self, component: str, device: str) -> float:
        """Get percentage for a specific component and device"""
        if self.device_config and component in self.device_config:
            return self.device_config[component].get(device, 0)
        else:
            # Fallback to legacy percent format
            if device == 'numa0':
                if component == 'weight':
                    return self.w_cpu_percent
                elif component == 'cache':
                    return self.cache_cpu_percent
                elif component == 'activation':
                    return self.act_cpu_percent
            elif device == 'disk':
                if component == 'weight':
                    return self.w_disk_percent
                elif component == 'cache':
                    return self.cache_disk_percent
                elif component == 'activation':
                    return self.act_disk_percent
            return 0
    
    @classmethod
    def create_from_device_config(cls, 
                                 batch_size: int,
                                 num_batches: int,
                                 device_config: dict,
                                 overlap: bool,
                                 sep_layer: bool,
                                 attn_sparsity: float,
                                 comp_weight: bool,
                                 comp_weight_config: Any,
                                 comp_cache: bool,
                                 comp_cache_config: Any):
        """Create Policy from new device configuration format"""
        # Extract legacy percentages for backward compatibility
        w_cpu_percent = device_config.get('weight', {}).get('numa0', 0)
        cache_cpu_percent = device_config.get('cache', {}).get('numa0', 0)
        act_cpu_percent = device_config.get('activation', {}).get('numa0', 0)
        
        return cls(
            batch_size=batch_size,
            num_batches=num_batches,
            w_cpu_percent=w_cpu_percent,
            cache_cpu_percent=cache_cpu_percent,
            act_cpu_percent=act_cpu_percent,
            overlap=overlap,
            sep_layer=sep_layer,
            attn_sparsity=attn_sparsity,
            comp_weight=comp_weight,
            comp_weight_config=comp_weight_config,
            comp_cache=comp_cache,
            comp_cache_config=comp_cache_config,
            device_config=device_config
        )
    
    
def cpu_mem_stats() -> int:
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.untyped_storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.untyped_storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None


def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def project_decode_latency(costs, prompt_len, gen_len):
    decode_costs = costs[1:]

    if gen_len / prompt_len < 0.1:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))
    else:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))

    return decode_latency


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
                        projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"projected: {projected}\n"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str

@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark results."""
    prefill_latency: float
    prefill_throughput: float
    decode_latency: float
    decode_throughput: float
    total_latency: float
    total_throughput: float


def read_benchmark_log(filename):
    with open(filename) as fin:
        lines = fin.readlines()

    def extract(line):
        a, b = line.split("\t")
        latency = a[a.index(":") + 1:a.index(" s")]
        throughput = b[b.index(":") + 1:b.index(" to")]
        return float(latency), float(throughput)

    prefill_latency, prefill_throughput = extract(lines[2])
    decode_latency, decode_throughput = extract(lines[3])
    total_latency, total_throughput = extract(lines[4])

    return BenchmarkResult(
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput,
    )