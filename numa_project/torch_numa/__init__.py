"""
PyTorch NUMA Extension
提供 NUMA 感知的内存分配功能
"""

import torch
from typing import List, Union, Optional

# 导入我们的 C++ 扩展
try:
    # 直接导入 C++ 扩展模块
    import torch_numa_cpp as _torch_numa_cpp
    
    # 重新导出主要函数
    get_numa_nodes = _torch_numa_cpp.get_numa_nodes
    get_current_numa_node = _torch_numa_cpp.get_current_numa_node
    bind_to_numa_node = _torch_numa_cpp.bind_to_numa_node
    register_numa_allocator = _torch_numa_cpp.register_numa_allocator
    set_default_numa_node = _torch_numa_cpp.set_default_numa_node
    get_tensor_numa_node = _torch_numa_cpp.get_tensor_numa_node

    # 张量操作函数
    _tensor_on_node = _torch_numa_cpp._tensor_on_node
    to = _torch_numa_cpp.to

    # 向后兼容
    create_tensor_on_node = _torch_numa_cpp.create_tensor_on_node
    migrate_tensor_to_node = _torch_numa_cpp.migrate_tensor_to_node

    def empty(*size, 
              node: int = 0, 
              dtype: torch.dtype = torch.float32, 
              device: str = 'cpu',
              requires_grad: bool = False, 
              **kwargs) -> torch.Tensor:
        """
        在指定 NUMA 节点创建空张量（类似 torch.empty）
        
        Args:
            *size: 张量形状，可以是可变参数或单个序列
                   例如：empty(3, 4) 或 empty([3, 4]) 或 empty((3, 4))
            node: NUMA 节点 ID
            dtype: 数据类型
            device: 设备类型（目前只支持 'cpu'）
            requires_grad: 是否需要梯度
            
        Returns:
            在指定 NUMA 节点的空张量
            
        Example:
            >>> tensor = torch_numa.empty(10, 10, node=1)
            >>> tensor = torch_numa.empty([100, 100], node=0, dtype=torch.float64)
            >>> tensor = torch_numa.empty((5, 5, 5), node=1)
        """
        # 处理 PyTorch 风格的大小参数
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            # 如果传入的是 empty([3, 4]) 或 empty((3, 4))
            shape = list(size[0])
        elif len(size) == 1 and isinstance(size[0], int):
            # 如果传入的是 empty(10)
            shape = [size[0]]
        elif all(isinstance(s, int) for s in size):
            # 如果传入的是 empty(3, 4, 5)
            shape = list(size)
        else:
            raise ValueError(f"Invalid size argument: {size}")
        
        tensor = _tensor_on_node(shape, node, dtype)
        
        if requires_grad:
            tensor.requires_grad_(True)
            
        return tensor

    def zeros(*size, 
              node: int = 0, 
              dtype: torch.dtype = torch.float32,
              requires_grad: bool = False, 
              **kwargs) -> torch.Tensor:
        """在指定 NUMA 节点创建零张量（类似 torch.zeros）"""
        tensor = empty(*size, node=node, dtype=dtype, requires_grad=requires_grad, **kwargs)
        tensor.zero_()
        return tensor

    def ones(*size, 
             node: int = 0, 
             dtype: torch.dtype = torch.float32,
             requires_grad: bool = False, 
             **kwargs) -> torch.Tensor:
        """在指定 NUMA 节点创建全一张量（类似 torch.ones）"""
        tensor = empty(*size, node=node, dtype=dtype, requires_grad=requires_grad, **kwargs)
        tensor.fill_(1.0)
        return tensor

    def randn(*size, 
              node: int = 0, 
              dtype: torch.dtype = torch.float32,
              requires_grad: bool = False, 
              **kwargs) -> torch.Tensor:
        """在指定 NUMA 节点创建随机正态分布张量（类似 torch.randn）"""
        tensor = empty(*size, node=node, dtype=dtype, requires_grad=requires_grad, **kwargs)
        tensor.normal_(0, 1)
        return tensor

    def rand(*size, 
             node: int = 0, 
             dtype: torch.dtype = torch.float32,
             requires_grad: bool = False, 
             **kwargs) -> torch.Tensor:
        """在指定 NUMA 节点创建随机均匀分布张量（类似 torch.rand）"""
        tensor = empty(*size, node=node, dtype=dtype, requires_grad=requires_grad, **kwargs)
        tensor.uniform_(0, 1)
        return tensor

    # 为 torch.Tensor 添加 NUMA 方法
    def _numa_to(self, node: int) -> torch.Tensor:
        """为 torch.Tensor 添加的 numa_to 方法"""
        return to(self, node)

    def _numa_node(self) -> int:
        """为 torch.Tensor 添加的 numa_node 属性"""
        return get_tensor_numa_node(self)

    # 动态添加方法
    torch.Tensor.numa_to = _numa_to
    torch.Tensor.numa_node = property(_numa_node)

    __version__ = "0.0.1"
    __all__ = [
        'empty', 'zeros', 'ones', 'randn', 'rand', 'to',
        'get_numa_nodes', 'get_current_numa_node', 'bind_to_numa_node',
        'register_numa_allocator', 'set_default_numa_node', 'get_tensor_numa_node',
        'create_tensor_on_node', 'migrate_tensor_to_node'
    ]

except ImportError as e:
    print(f"Warning: Failed to import torch_numa_cpp: {e}")
    print("Please ensure the C++ extension is properly compiled.")
    raise ImportError(f"torch_numa extension not available: {e}")