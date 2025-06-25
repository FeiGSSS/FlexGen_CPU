#!/usr/bin/env python3
"""
PyTorch NUMA Extension 基本功能测试
"""

import sys

def test_wrapper_import():
    """测试包装器导入功能"""
    try:
        import torch_numa_wrapper
        print("✓ torch_numa_wrapper 导入成功")
        return True
    except ImportError as e:
        print(f"✗ torch_numa_wrapper 导入失败: {e}")
        return False

def test_basic_functions():
    """测试基本功能"""
    try:
        import torch_numa
        
        # 测试节点查询
        nodes = torch_numa.get_numa_nodes()
        print(f"✓ NUMA 节点数: {nodes}")
        
        # 测试当前节点查询
        current = torch_numa.get_current_numa_node()
        print(f"✓ 当前节点: {current}")
        
        return True
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_tensor_operations():
    """测试张量操作"""
    try:
        import torch_numa
        
        nodes = torch_numa.get_numa_nodes()
        if nodes == 0:
            print("! 系统不支持 NUMA，跳过张量操作测试")
            return True
        
        # 测试张量创建
        tensor = torch_numa.create_tensor_on_node([10, 10], 0)
        print("✓ 张量创建成功")
        
        # 测试节点查询
        node = torch_numa.get_tensor_numa_node(tensor)
        print(f"✓ 张量节点查询: {node}")
        
        # 测试张量迁移
        if nodes > 1:
            migrated = torch_numa.migrate_tensor_to_node(tensor, min(1, nodes-1))
            new_node = torch_numa.get_tensor_numa_node(migrated)
            print(f"✓ 张量迁移成功: {new_node}")
        
        return True
    except Exception as e:
        print(f"✗ 张量操作测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== PyTorch NUMA Extension 测试 ===")
    print()
    
    tests = [
        ("包装器导入测试", test_wrapper_import),
        ("基本功能测试", test_basic_functions),
        ("张量操作测试", test_tensor_operations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"运行 {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} 执行错误: {e}")
            failed += 1
        print()
    
    print("=== 测试结果 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")
    
    if failed == 0:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
