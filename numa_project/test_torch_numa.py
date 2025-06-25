#!/usr/bin/env python3
"""
PyTorch NUMA 扩展的 Torch 集成测试
测试与 PyTorch 核心功能的集成
"""

def test_torch_numa_integration():
    """测试 torch_numa 与 PyTorch 的集成"""
    try:
        import torch
        import torch_numa
        
        print("=== Torch NUMA 集成测试 ===")
        
        # 基本信息
        print(f"PyTorch 版本: {torch.__version__}")
        nodes = torch_numa.get_numa_nodes()
        print(f"NUMA 节点数: {nodes}")
        
        if nodes == 0:
            print("警告: 系统不支持 NUMA")
            return True
        
        # 注册 NUMA 分配器
        torch_numa.register_numa_allocator()
        print("NUMA 分配器注册成功")
        
        # 创建张量测试
        tensor = torch.randn(100, 100)
        print(f"创建张量: {tensor.shape}")
        
        # 检查张量节点
        node = torch_numa.get_tensor_numa_node(tensor)
        print(f"张量所在节点: {node}")
        
        # 在特定节点创建张量
        if nodes > 0:
            numa_tensor = torch_numa.create_tensor_on_node([50, 50], 0)
            numa_node = torch_numa.get_tensor_numa_node(numa_tensor)
            print(f"NUMA 张量节点: {numa_node}")
        
        print("集成测试通过！")
        return True
        
    except Exception as e:
        print(f"集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    if test_torch_numa_integration():
        sys.exit(0)
    else:
        sys.exit(1)
