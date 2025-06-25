from __future__ import annotations

import os
import shutil
import queue
import threading
import sys

from enum import Enum, auto
from typing import Union, Tuple, List, Any, Dict
from itertools import  count

import numpy as np
import torch

# NUMA support
NUMA_AVAILABLE = False
NUMA_NODES = 0
torch_numa = None

def init_numa_support():
    """Initialize NUMA support if available"""
    global NUMA_AVAILABLE, NUMA_NODES, torch_numa
    try:
        numa_project_path = os.path.join(os.path.dirname(__file__), '..', 'numa_project')
        if numa_project_path not in sys.path:
            sys.path.insert(0, numa_project_path)
        import torch_numa as _torch_numa
        torch_numa = _torch_numa
        NUMA_AVAILABLE = True
        NUMA_NODES = torch_numa.get_numa_nodes()
        torch_numa.register_numa_allocator()
        print(f"NUMA support enabled with {NUMA_NODES} nodes")
        return True
    except Exception as e:
        print(f"NUMA support not available: {e}")
        NUMA_AVAILABLE = False
        NUMA_NODES = 0
        return False

from flexgen.utils import (
    Policy,
    Task,
    cpu_mem_stats,
    torch_dtype_to_num_bytes,
    torch_dtype_to_np_dtype,
    np_dtype_to_torch_dtype,
    GB
)
from flexgen.opt_config import OptConfig

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None

def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice

class DeviceType(Enum):
    # NUMA节点 - 动态生成
    NUMA0 = auto()
    NUMA1 = auto() 
    NUMA2 = auto()
    NUMA3 = auto()
    NUMA4 = auto()
    NUMA5 = auto()
    NUMA6 = auto()
    NUMA7 = auto()
    # 其他设备类型
    DISK = auto()
    MIXED = auto()  # For TorchMixedDevice, which manages multiple devices
    COMPRESSED = auto()

    @staticmethod
    def convert(name: str):
        """Convert string name to DeviceType"""
        name = name.lower()
        if name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        elif name.startswith("numa"):
            try:
                numa_id = int(name[4:])  # Extract number from "numa0", "numa1", etc.
                return getattr(DeviceType, f"NUMA{numa_id}")
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid NUMA device name: {name}")
        # 向后兼容：将原来的"cpu"映射到numa0
        elif name == "cpu":
            print("Warning: 'cpu' device type is deprecated, using 'numa0' instead")
            return DeviceType.NUMA0
        else:
            raise ValueError(f"Unknown device type: {name}")
    
    @staticmethod
    def get_available_numa_devices():
        """Get list of available NUMA device types based on system"""
        if not NUMA_AVAILABLE:
            return [DeviceType.NUMA0]  # Fallback to single node
        
        available = []
        for i in range(min(NUMA_NODES, 8)):  # Max 8 NUMA nodes supported
            available.append(getattr(DeviceType, f"NUMA{i}"))
        return available
    
    def to_numa_node_id(self) -> int:
        """Convert DeviceType to NUMA node ID"""
        if self.name.startswith("NUMA"):
            return int(self.name[4:])
        else:
            raise ValueError(f"Not a NUMA device type: {self}")
    
    def is_numa_device(self) -> bool:
        """Check if this is a NUMA device type"""
        return self.name.startswith("NUMA")


class TorchTensor():
    name_count = count()
    def __init__(
        self,
        shape: Tuple,
        dtype: torch.dtype,
        data: Any,
        device: Union[TorchDevice, TorchDisk, TorchMixedDevice],
        name: str = None
    ):
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device
        self.name = name or TorchTensor.next_name()
        
        if self.device.DeviceType == DeviceType.DISK:
            self.delete_file = True

    
    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]
    
    @classmethod
    def next_name(cls) -> str:
        return f"t_{next(cls.name_count)}"
    
    @classmethod
    def create_from_torch(
        cls,
        data: torch.Tensor,
        device: Union[TorchDevice, TorchDisk, TorchMixedDevice],
        name: str = None
    ):
        return cls(data.shape, data.dtype, data, device, name)
    
    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.DeviceType == DeviceType.DISK:
            self.device.delete(self)
        elif self.device.DeviceType == DeviceType.MIXED:
            self.device.delete(self)
        self.device = self.data = None
            
    def load_from_np(self, np_array:np.ndarray) -> None:
        if self.device.DeviceType == DeviceType.DISK:
            with open(self.data, "wb") as f:
                np.save(f, np_array)
        else:
            if self.device.DeviceType == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))
    
    def load_from_np_file(self, file_name: str):
        if self.device.DeviceType == DeviceType.DISK:
            shutil.copy(file_name, self.data)
        else:
            self.load_from_np(np.load(file_name))
    
    def copy(self,
             dst: Union[TorchDevice, TorchDisk, TorchMixedDevice],
             src_indices: Tuple[slice] = None) -> TorchTensor:
        """
        Copy the tensor to a new device or disk.
        """
        if src_indices:
            assert all(x.step is None for x in src_indices), "Only support slice with step=1"
            shape = tuple(x.stop - x.start for x in src_indices) + self.shape[len(src_indices):]
        else:
            shape = self.shape
        if dst.DeviceType == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret
    
    def smart_copy(self, dst:Union[TorchDevice, TorchDisk, TorchMixedDevice],
                   src_indices: Tuple[slice] = None) -> Tuple[TorchTensor, bool]:
        """
        Smart copy the tensor to a new device or disk.
        """
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices), True

    def move(self, dst: Union[TorchDevice, TorchDisk, TorchMixedDevice]) -> TorchTensor:
        """
        Move the tensor to a new device or disk.
        """
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret
    
    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")

########### TorchDevice ###########

class TorchDevice:
    """Wrap tensor APIs of a single device (NUMA node or traditional CPU)"""
    def __init__(self, name: str, mem_capacity: int = None, flops=None, numa_node: int = None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops
        self.numa_node = numa_node
        
        # Ensure recursive imports are resolved
        if TorchCompressedDevice is None:
            fix_recursive_import()
        
        # Handle different device types
        if numa_node is not None:
            # Explicit NUMA node specified
            self.dev = torch.device("cpu")
            self.DeviceType = getattr(DeviceType, f"NUMA{numa_node}")
        elif name.startswith("numa"):
            # NUMA device specified by name
            self.DeviceType = DeviceType.convert(name)
            self.numa_node = self.DeviceType.to_numa_node_id()
            self.dev = torch.device("cpu")
        elif name == "cpu":
            # Legacy CPU - convert to NUMA0 with warning
            self.DeviceType = DeviceType.convert(name)  # This will show warning
            self.numa_node = 0
            self.dev = torch.device("cpu")
        else:
            # Non-NUMA device (disk, etc.)
            self.dev = torch.device(name)
            self.DeviceType = DeviceType.convert(name)
            self.numa_node = None
        
        # Initialize NUMA support if needed
        if self.DeviceType.is_numa_device() and not NUMA_AVAILABLE:
            init_numa_support()
        
        self.compressed_device = TorchCompressedDevice(self)
        
        self.attention_compute_workspace = None
        self.workspace_pointer = None
        
        # Set global CPU device for compatibility (use NUMA0)
        if self.DeviceType == DeviceType.NUMA0:
            global global_cpu_device
            global_cpu_device = self
        
    def allocate(self,
                 shape: Tuple,
                 dtype: np.dtype,
                 name: str = None) -> TorchTensor:        
        dtype = np_dtype_to_torch_dtype[dtype]
        
        # Use NUMA-aware allocation if this is a NUMA device
        if self.DeviceType.is_numa_device() and NUMA_AVAILABLE and torch_numa is not None:
            data = torch_numa.empty(*shape, node=self.numa_node, dtype=dtype)
        else:
            # Fallback to regular allocation
            data = torch.empty(shape, dtype=dtype, device=self.dev)
            
        return TorchTensor.create_from_torch(data, self, name)
    
    
    def init_attention_compute_workspace(self,
                                         config: OptConfig,
                                         task: Task,
                                         policy: Policy) -> None:
        if not self.DeviceType.is_numa_device():
            return  # Only NUMA devices require this fp32 workspace
        
        if not policy.comp_cache:
            batch_size = policy.batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pointer = 0
            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, batch_size * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32)
                v_cache = self.allocate(shape, np.float32)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(config, task, policy)

    def next_attention_compute_workspace(self) -> Tuple[TorchTensor, TorchTensor]:
        self.workspace_pointer = (self.workspace_pointer + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pointer]
    
    def del_attention_compute_workspace(self) -> None:
        self.attention_compute_workspace = None
        self.workspace_pointer = None
        
    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self,
                              attention_mask: TorchTensor,
                              donate: List[bool]) -> TorchTensor:
        b = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
                             torch.ones((b, 1), dtype=attention_mask.dtype, device=self.dev)),
                            dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)
        
    def init_cache_one_batch(self,
                             config: OptConfig,
                             task: Task,
                             policy: Policy) -> Tuple[TorchTensor, TorchTensor]:
        n_head = config.n_head
        hidden_size = config.input_dim
        prompt_len = task.prompt_len
        gen_len = task.gen_len
        batch_size = policy.batch_size
        
        shape = (prompt_len + gen_len - 1, batch_size * n_head, hidden_size // n_head)
        k_cache = self.allocate(shape, np.float32)
        v_cache = self.allocate(shape, np.float32)

        return k_cache, v_cache
    
    def delete(self, tensor: TorchTensor) -> None:
        pass
    
    def mem_stats(self) -> Tuple[int, int]:
        if self.DeviceType.is_numa_device():
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()
        
        return cur_mem, peak_mem
    
    def print_stats(self, output_file=None):
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


################# TorchDisk #################


class TorchDisk:
    """Manage tensors stored on a disk."""
    def __init__(self,
                 path: str,
                 mem_capacity: int = None):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity
        
        from flexgen.compression import TorchCompressedDevice
        self.compressed_device = TorchCompressedDevice(self)
        
        self.DeviceType: DeviceType = DeviceType.DISK
        
    def allocate(self,
                 shape: Tuple,
                 dtype: np.dtype,
                 name: str = None) -> TorchTensor:
        """
        Allocate a tensor on disk.
        """
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor: TorchTensor) -> None:
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)
            
    def init_cache_one_batch(
        self,
        config: OptConfig,
        task: Task,
        policy: Policy
    ) -> TorchTensor:
        """
        Initialize a cache for one batch on disk.
        """
        n_head = config.n_head
        hidden_size = config.input_dim
        prompt_len = task.prompt_len
        gen_len = task.gen_len
        batch_size = policy.batch_size
        
        shape = (prompt_len + gen_len - 1, batch_size * n_head, hidden_size // n_head)
        k_cache = self.allocate(shape, np.float32)
        v_cache = self.allocate(shape, np.float32)
        return k_cache, v_cache
    
    def mem_stats(self):
        raise NotImplementedError("Disk memory stats not implemented")
    
    def print_stats(self, output_file=None):
        raise NotImplementedError("Disk print stats not implemented")
    
    
# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.DeviceType = DeviceType.MIXED
        self.base_devices = base_devices
        
    def allocate(self,
                 shape: Tuple,
                 dtype: np.dtype,
                 seg_lengths: List,
                 name: str = None) -> TorchTensor:
        
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)
        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype))
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor: TorchTensor) -> None:
        for x in tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_batch(self,
                             config: OptConfig,
                             task: Task,
                             policy: Policy) -> Tuple[TorchTensor, TorchTensor]:
        num_head = config.n_head
        hidden_size = config.input_dim
        prompt_len = task.prompt_len
        gen_len = task.gen_len
        batch_size = policy.batch_size
        shape = (prompt_len + gen_len - 1, batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_cpu = shape[SEG_DIM]
            len_disk = 0
        else:
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_cpu
        lens = [len_cpu, len_disk]

        k_cache = self.allocate(shape, np.float32, seg_lengths=lens)
        v_cache = self.allocate(shape, np.float32, seg_lengths=lens)
        return k_cache, v_cache


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    # assert dst.device != src.device, "Source and destination must be different devices, NO COPY if in the same device"
    # Currently only support DISK -> DISK or CPU -> DISK copy
    async_io_manager = AsyncIOManager() # Sigleton instance
    if dst.device.DeviceType == DeviceType.MIXED:
        assert src.device.DeviceType != DeviceType.MIXED
        seg_points = dst.data[1]
        
        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1], base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    
    elif src.device.DeviceType == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.DeviceType != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.DeviceType == DeviceType.COMPRESSED or
          dst.device.DeviceType == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    else:
        # print(dst, src)
        async_io_manager.submit_copy(dst, dst_indices, src, src_indices)

class AsyncIOManager:
    """
    Asynchronous I/O manager for tensor operations using threading.
    Handles CPU <-> Disk copies.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncIOManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, num_threads: int = 4):
        if self._initialized:
            # Initialize instance only once
            return
        
        self.num_threads = num_threads
        self.copy_queue = queue.Queue()
        self.copy_threads = []
        
        for i in range(self.num_threads):
            t = threading.Thread(
                target=self._copy_worker_func,
                name=f"AsyncIOCopyThread-{i}" # Naming threads is good for debugging
            )
            t.daemon = True  # Allow threads to exit when the main program exits
            t.start()
            self.copy_threads.append(t)
        
        self._initialized = True
        self._closed = False

    def submit_copy(self,
                    dst: TorchTensor,
                    dst_indices, 
                    src: TorchTensor,
                    src_indices):
        """Submit an asynchronous copy task to the queue."""
        if self._closed:
            raise RuntimeError("AsyncIOManager has been closed.")
        # # print(dst, src)
        self.copy_queue.put_nowait((dst, dst_indices, src, src_indices))

    def synchronize(self):
        """Wait for all submitted copy tasks in the queue to complete."""
        if not self._closed and self.copy_queue:
            self.copy_queue.join()
    
    def close(self):
        for _ in range(self.num_threads):
            self.copy_queue.put_nowait(None)  # Send sentinel to stop threads
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None
        self._closed = True
    
    def map_to_torch_tensor(self, tensor: TorchTensor, indices):
        if tensor.device.DeviceType == DeviceType.DISK:
            data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
        else:
            data = tensor.data

        return data[indices] if indices else data
        
    def _copy_worker_func(self):
        """Worker function that processes copy tasks from the queue."""
        while True:
            item = self.copy_queue.get() # Blocks until an item is available
            if item is None: # Sentinel for thread termination
                self.copy_queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            # # print(dst, src)
            src_data = self.map_to_torch_tensor(src, src_indices)
            dst_data = self.map_to_torch_tensor(dst, dst_indices)
            
            dst_data.copy_(src_data) # Copy data from source to destination
            self.copy_queue.task_done() # Mark the task as done
    
    def __del__(self):
        # Ensure threads are cleaned up if the manager object is deleted
        if self.copy_queue:
            self.close()