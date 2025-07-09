from __future__ import annotations

import os
import shutil
import queue
import threading
import ctypes
from concurrent.futures import ThreadPoolExecutor

from enum import Enum, auto
from typing import Union, Tuple, List, Any
from itertools import  count

import numpy as np
import torch

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
from flexgen.libnuma_wrapper import libnuma

def fix_recursive_import():
    pass

class OptimizedMemCopy:
    def __init__(self):
        self.enable_optimization = True
        self.parallel_threshold = 64 * 1024 * 1024
        self.num_threads = 16
        self.stats = {
            'total_copies': 0,
            'optimized_copies': 0,
            'total_bytes': 0,
            'optimized_bytes': 0
        }
    
    def optimized_memmove(self,
                          dst_ptr: int,
                          src_ptr: int,
                          size: int):
        self.stats['total_copies'] += 1
        self.stats['total_bytes'] += size
        
        if (not self.enable_optimization or 
            size < self.parallel_threshold):
            ctypes.memmove(ctypes.c_void_p(dst_ptr), 
                          ctypes.c_void_p(src_ptr), 
                          ctypes.c_size_t(size))
            return
        
        self.stats['optimized_copies'] += 1
        self.stats['optimized_bytes'] += size
        
        num_threads = min(self.num_threads, os.cpu_count())
        
        chunk_size = size // num_threads
        
        def copy_chunk(thread_id):
            start_offset = thread_id * chunk_size
            if thread_id == num_threads - 1:
                copy_size = size - start_offset
            else:
                copy_size = chunk_size
            
            dst_chunk = int(dst_ptr) + int(start_offset)
            src_chunk = int(src_ptr) + int(start_offset)
            
            if dst_chunk < 0 or src_chunk < 0:
                raise ValueError(f"Invalid pointer calculation: dst={dst_chunk}, src={src_chunk}")
            
            ctypes.memmove(ctypes.c_void_p(dst_chunk), 
                          ctypes.c_void_p(src_chunk), 
                          ctypes.c_size_t(copy_size))
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(copy_chunk, i) for i in range(num_threads)]
            for future in futures:
                future.result()
    
    def get_stats(self):
        return self.stats.copy()
    
    def print_stats(self):
        stats = self.get_stats()
        opt_ratio = (stats['optimized_copies'] / max(stats['total_copies'], 1)) * 100
        opt_bytes_ratio = (stats['optimized_bytes'] / max(stats['total_bytes'], 1)) * 100
        
        print(f"Memory Copy Optimization Stats:")
        print(f"  Total copies: {stats['total_copies']}")
        print(f"  Optimized copies: {stats['optimized_copies']} ({opt_ratio:.1f}%)")
        print(f"  Total bytes: {stats['total_bytes'] / (1024**3):.2f} GB")
        print(f"  Optimized bytes: {stats['optimized_bytes'] / (1024**3):.2f} GB ({opt_bytes_ratio:.1f}%)")

_optimized_copy = OptimizedMemCopy()
_optimized_copy.enable_optimization = True

def print_memory_copy_stats():
    _optimized_copy.print_stats()

class DeviceType(Enum):
    CPU = auto()
    NUMA = auto()
    DISK = auto()

    @staticmethod
    def convert(name: str):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "numa":
            return DeviceType.NUMA
        elif name == "disk":
            return DeviceType.DISK
        else:
            raise NotImplementedError(f"DeviceType {name} not implemented")
        
class TorchTensor():
    name_count = count()
    def __init__(
        self,
        shape: Tuple,
        dtype: torch.dtype,
        data: Any,
        device: Union[TorchDevice, TorchDisk, TorchNuma],
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
        device: Union[TorchDevice, TorchDisk, TorchNuma],
        name: str = None
    ):
        return cls(data.shape, data.dtype, data, device, name)
    
    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.DeviceType == DeviceType.DISK:
            self.device.delete(self)
        elif self.device.DeviceType == DeviceType.NUMA:
            self.device.delete(self)
        self.device = self.data = None
            
    def load_from_np(self, np_array:np.ndarray) -> None:
        if self.device.DeviceType == DeviceType.DISK:
            with open(self.data, "wb") as f:
                np.save(f, np_array)
        elif self.device.DeviceType == DeviceType.NUMA:
            # Load data from a NumPy array into a pre-allocated NUMA memory region.
            # This involves a direct memory copy from the source array to the target pointer.
            ptr, byte_size, shape, dtype = self.data
            # --- Pre-copy validation ---
            if ptr is None or ptr == 0:
                raise ValueError("Cannot load data into an invalid (null) NUMA pointer.")
            if np_array.shape != shape:
                raise ValueError(f"Shape mismatch for tensor '{self.name}': expected {shape}, but got {np_array.shape}.")
            if np_array.dtype != dtype:
                raise ValueError(f"Dtype mismatch for tensor '{self.name}': expected {dtype}, but got {np_array.dtype}.")
            if byte_size != np_array.nbytes:
                raise ValueError(f"Byte size mismatch for tensor '{self.name}': expected {byte_size}, but got {np_array.nbytes}.")

            # Ensure the source NumPy array has a C-style contiguous memory layout,
            # which is required for a safe and correct single memmove operation.
            if not np_array.flags.c_contiguous:
                print(f"DEBUG: Source np.ndarray for '{self.name}' is not C-contiguous. Creating a contiguous copy.")
                np_array = np.ascontiguousarray(np_array)

            try:
                _optimized_copy.optimized_memmove(
                    dst_ptr=int(ptr),
                    src_ptr=int(np_array.ctypes.data),
                    size=byte_size
                )
            except Exception as e:
                # Raise an exception with detailed context for easier debugging.
                raise RuntimeError(
                    f"Failed to copy data to NUMA node for tensor '{self.name}': {e}. "
                    f"Destination Pointer: {hex(ptr)}, Source Pointer: {hex(np_array.ctypes.data)}, "
                    f"Bytes to copy: {byte_size}"
                ) from e
        else:
            self.data.copy_(torch.from_numpy(np_array))
    
    def load_from_np_file(self, file_name: str):
        if self.device.DeviceType == DeviceType.DISK:
            shutil.copy(file_name, self.data)
        else:
            self.load_from_np(np.load(file_name))
    
    def copy(self,
             dst_device: Union[TorchDevice, TorchDisk, TorchNuma],
             src_indices: Tuple[slice] = None) -> TorchTensor:
        """
        Copy the tensor to a new device or disk.
        """
        if src_indices:
            raise NotImplementedError("Slicing is not supported in TorchTensor.copy()")
        dst = dst_device.allocate(self.shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(dst, None, self, src_indices)
        return dst

    def smart_copy(self, dst_device, src_indices = None) -> Tuple[TorchTensor, bool]:
        """
        Smart copy the tensor to a new device or disk.
        """
        if self.device.DeviceType == dst_device.DeviceType:
            return self, False
        return self.copy(dst_device, src_indices), True

    def move(self, dst_device: Union[TorchDevice, TorchDisk, TorchNuma]) -> TorchTensor:
        """
        Move the tensor to a new device or disk.
        """
        if self.device.DeviceType == dst_device.DeviceType:
            return self
        ret = self.copy(dst_device)
        self.delete()
        return ret
    
    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")

########### TorchDevice ###########

class TorchDevice:
    """Wrap tensor APIs of a single CPU"""
    def __init__(self, name='cpu'):
        self.name = name
        self.dev = torch.device(name)
        self.DeviceType: DeviceType = DeviceType.convert(self.dev.type)
        
        self.attention_compute_workspace = None
        self.workspace_pointer = None

        
    def allocate(self,
                 shape: Tuple,
                 dtype: np.dtype,
                 name: str = None) -> TorchTensor:   
        name = name or TorchTensor.next_name()   
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name)
    
    
    def init_attention_compute_workspace(self,
                                         config: OptConfig,
                                         task: Task,
                                         policy: Policy) -> None:
        if self.DeviceType != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace
        
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
        if self.DeviceType == DeviceType.CPU:
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
    def __init__(self, path: str):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self._meta_data = {}
        self.DeviceType: DeviceType = DeviceType.DISK
        
        self.dev = None
        
    def allocate(self,
                 shape: Tuple,
                 dtype: np.dtype,
                 name: str = None) -> TorchTensor:
        """
        Allocate a tensor on disk.
        """
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        self._meta_data[name] = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor: TorchTensor) -> None:
        if tensor.name in self._meta_data and tensor.delete_file:
            path = self._meta_data.pop(tensor.name)
            if os.path.exists(path):
                os.remove(path)
            
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
    
    def __del__(self):
        if not hasattr(self, '_meta_data'):
            return
        try:
            keys = list(self._meta_data.keys())
            for key in keys:
                path = self._meta_data.pop(key)
                if os.path.exists(path):
                    os.remove(path)
        except Exception:
            pass
        
    
################# TorchNuma #################
class TorchNuma:
    """Manage tensors stored on a NUMA node."""
    def __init__(self, numa_node: int=2):
        self.numa_node = numa_node
        self._metadata = {}  # {key: (ptr_as_int, byte_size, shape, dtype)}
        self._lock = threading.Lock()
        self.DeviceType: DeviceType = DeviceType.NUMA
        self.dev = None
        
    def allocate(self,
                 shape: Tuple,
                 dtype: np.dtype,
                 name: str = None) -> TorchTensor:
        """
        Allocate a tensor on a NUMA node.
        """
        name = name or TorchTensor.next_name()
        byte_size = np.prod(shape) * torch_dtype_to_num_bytes[np_dtype_to_torch_dtype[dtype]]
        ptr = libnuma.alloc_onnode(byte_size, self.numa_node)
        if ptr is None:
            raise MemoryError(f"Failed to allocate {byte_size} bytes on NUMA node {self.numa_node}")
        self._metadata[name] = (ptr, byte_size, shape, dtype)
        
        return TorchTensor(shape,
                           np_dtype_to_torch_dtype[dtype],
                           (ptr, byte_size, shape, dtype),
                           self,
                           name=name)
        

    def delete(self, tensor: TorchTensor, internal_call=False) -> None:
        def _free_op():
            if tensor.name in self._metadata:
                ptr, byte_size, _, _ = self._metadata.pop(tensor.name)
                libnuma.free(ptr, byte_size)
        if internal_call:
            _free_op()
        else:
            with self._lock:
                # Ensure thread safety when freeing memory
                _free_op()
        
            
    def init_cache_one_batch(
        self,
        config: OptConfig,
        task: Task,
        policy: Policy) -> TorchTensor:
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
        raise NotImplementedError("NUMA memory stats not implemented")
    
    def print_stats(self, output_file=None):
        raise NotImplementedError("NUMA print stats not implemented")

    def __del__(self):
        if not hasattr(self, '_metadata') or not hasattr(self, '_lock'):
            return
        try:
            with self._lock: # Ensure thread safety when freeing memory
                keys = list(self._metadata.keys())
                for key in keys:
                    ptr, byte_size, _, _ = self._metadata.pop(key)
                    libnuma.free(ptr, byte_size)
        except Exception:
            pass
    
    
SEG_DIM = 1  # The dimension of the segment in the indices tuple

def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    async_io_manager = AsyncIOManager() # Sigleton instance
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

    def __init__(self, num_threads = 4):
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
        if isinstance(data, torch.Tensor) and indices is not None:
            data = data[indices]
        return data
        
    def _copy_worker_func(self):
        while True:
            item = self.copy_queue.get()
            if item is None:
                self.copy_queue.task_done()
                return

            dst, dst_indices, src, src_indices = item

            src_data = self.map_to_torch_tensor(src, src_indices)
            dst_data = self.map_to_torch_tensor(dst, dst_indices)
            
            if src.device.DeviceType == DeviceType.NUMA:
                # NUMA -> Other device
                self._copy_from_numa(src_data, src_indices, dst_data)
            elif dst.device.DeviceType == DeviceType.NUMA:
                # Other device -> NUMA
                self._copy_to_numa(dst_data, dst_indices, src_data)
            else:
                # Normal tensor copy
                dst_data.copy_(src_data)
            
            self.copy_queue.task_done()

    def _copy_from_numa(self, src_data, src_indices, dst_data):
        """
        Copy data from a NUMA memory region to another device or disk.
        """
        base_ptr, total_byte_size, full_shape, src_dtype = src_data
        assert dst_data.dtype == np_dtype_to_torch_dtype[src_dtype]
        
        if src_indices is None:
            # Copy the entire NUMA tensor
            src_ptr = base_ptr
            copy_bytes = total_byte_size
        else:
            # Calculate the memory offset for the given indices
            src_ptr, copy_bytes = self._calculate_numa_slice_offset(
                base_ptr, full_shape, src_dtype, src_indices
            )

        # Verify size match
        expected_bytes = dst_data.numel() * dst_data.element_size()
        assert copy_bytes == expected_bytes, f"Size mismatch: {copy_bytes} != {expected_bytes}"
        
        if not dst_data.is_contiguous():
            dst_data = dst_data.contiguous()
            
        _optimized_copy.optimized_memmove(
            dst_ptr=int(dst_data.data_ptr()),
            src_ptr=int(src_ptr),
            size=copy_bytes
        )

    def _copy_to_numa(self, dst_data, dst_indices, src_data):
        """Copy data from another device to NUMA memory region."""
        base_ptr, total_byte_size, full_shape, dst_dtype = dst_data

        assert src_data.dtype == np_dtype_to_torch_dtype[dst_dtype]
        
        if dst_indices is None:
            assert src_data.shape == full_shape, \
                f"Shape mismatch: {src_data.shape} != {full_shape}"
            dst_ptr = base_ptr
            copy_bytes = total_byte_size
        else:
            # Calculate the memory offset for the given indices
            dst_ptr, copy_bytes = self._calculate_numa_slice_offset(
                base_ptr, full_shape, dst_dtype, dst_indices
            )
        
        # Verify size match
        expected_bytes = src_data.numel() * src_data.element_size()
        assert copy_bytes == expected_bytes, f"Size mismatch: {copy_bytes} != {expected_bytes}"
        
        if not src_data.is_contiguous():
            src_data = src_data.contiguous()
        try:
            _optimized_copy.optimized_memmove(
                dst_ptr=int(dst_ptr),
                src_ptr=int(src_data.data_ptr()),
                size=copy_bytes
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to copy data to NUMA node: {e}. "
                f"dst_ptr={dst_ptr}, copy_bytes={copy_bytes}, "
                f"src_data.shape={src_data.shape}, src_data.dtype={src_data.dtype}"
            ) from e

    def _calculate_numa_slice_offset(self, base_ptr: int, full_shape: tuple, 
                                     dtype: np.dtype, indices: tuple) -> tuple:
        """
        Calculate the memory offset and size of a slice in a NUMA Memory region.
        Only supports slicing on the first dimension: tensor[i:j, :, :, :]
        # ✅ 支持的模式
        tensor[10:20, :, :]       # 第一维切片，其他维度全选
        tensor[5, :, :, :]        # 第一维单个索引，其他维度全选
        tensor[10:20]             # 一维张量的切片

        # ❌ 不支持的模式
        tensor[:, 10:20, :]       # 第二维切片
        tensor[10:20, 5:10, :]    # 多维切片
        tensor[::2, :, :]         # 步长不为1
        """
        if len(indices) > len(full_shape):
            raise ValueError(f"Indices length {len(indices)} > shape length {len(full_shape)}")
        
        element_size = torch_dtype_to_num_bytes[np_dtype_to_torch_dtype[dtype]]
        
        # 验证只对第一个维度进行切片
        first_dim_idx = indices[0]
        
        # 检查第一个维度必须是slice或int
        if not isinstance(first_dim_idx, (slice, int)):
            raise ValueError(f"First dimension index must be slice or int, got {type(first_dim_idx)}")
        
        # 检查其他维度必须是完整切片或None（表示:）
        for i in range(1, len(indices)):
            idx = indices[i]
            if idx is None:
                continue  # None表示:，允许
            elif isinstance(idx, slice):
                # 必须是完整切片 [0:dim_size:1] 或 [:] 
                start = idx.start or 0
                stop = idx.stop if idx.stop is not None else full_shape[i]
                step = idx.step or 1
                
                if start != 0 or stop != full_shape[i] or step != 1:
                    raise ValueError(
                        f"Dimension {i} must be fully sliced (:), "
                        f"got slice({start}, {stop}, {step}) for shape[{i}]={full_shape[i]}"
                    )
            else:
                raise ValueError(
                    f"Dimension {i} must be fully sliced (:) or None, "
                    f"got {type(idx)}: {idx}"
                )
        
        # 计算第一个维度的偏移和大小
        if isinstance(first_dim_idx, int):
            # 单个索引：tensor[i, :, :, :]
            if first_dim_idx < 0:
                first_dim_idx += full_shape[0]  # 处理负索引
            if not (0 <= first_dim_idx < full_shape[0]):
                raise IndexError(f"Index {first_dim_idx} out of range for dimension 0 with size {full_shape[0]}")
            
            # 计算一个"行"的字节数（除第一维外的所有维度）
            row_elements = np.prod(full_shape[1:]) if len(full_shape) > 1 else 1
            row_bytes = row_elements * element_size
            
            offset_bytes = first_dim_idx * row_bytes
            slice_bytes = row_bytes
            
        elif isinstance(first_dim_idx, slice):
            # 切片索引：tensor[i:j, :, :, :]
            start = first_dim_idx.start or 0
            stop = first_dim_idx.stop if first_dim_idx.stop is not None else full_shape[0]
            step = first_dim_idx.step or 1
            
            if step != 1:
                raise NotImplementedError("Non-unit step not supported for NUMA slicing")
            
            # 处理负索引
            if start < 0:
                start += full_shape[0]
            if stop < 0:
                stop += full_shape[0]
                
            # 边界检查
            start = max(0, min(start, full_shape[0]))
            stop = max(start, min(stop, full_shape[0]))
            
            if start >= stop:
                raise ValueError(f"Invalid slice: start={start} >= stop={stop}")
            
            # 计算一个"行"的字节数
            row_elements = np.prod(full_shape[1:]) if len(full_shape) > 1 else 1
            row_bytes = row_elements * element_size
            
            offset_bytes = start * row_bytes
            slice_rows = stop - start
            slice_bytes = slice_rows * row_bytes
        
        return int(base_ptr + offset_bytes), int(slice_bytes)
    
    def __del__(self):
        # Ensure threads are cleaned up if the manager object is deleted
        if self.copy_queue:
            self.close()