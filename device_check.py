#!/usr/bin/env python3
"""
检查并测试 DeviceType 的使用情况
"""
import os
import sys
from enum import Enum, auto

# 添加项目目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flexgen.torch_backend import (
    DeviceType,
    TorchDevice,
    TorchDisk,
    TorchMixedDevice,
    fix_recursive_import,
    init_numa_support
)
from flexgen.utils import ExecutionEnv, Policy
import flexgen.torch_numa as torch_numa

# 确保递归导入被修复
fix_recursive_import()

# 初始化 NUMA 支持
init_numa_support()

def test_device_type_enum():
    """测试 DeviceType 枚举类型的行为"""
    print("测试 DeviceType 枚举：")
    print(f"NUMA0 = {DeviceType.NUMA0}")
    print(f"NUMA1 = {DeviceType.NUMA1}")
    print(f"NUMA2 = {DeviceType.NUMA2}")
    print(f"DISK = {DeviceType.DISK}")
    print(f"MIXED = {DeviceType.MIXED}")
    print(f"COMPRESSED = {DeviceType.COMPRESSED}")
    
    # 测试字符串转换
    print("\n测试字符串转换：")
    for s in ["numa0", "numa1", "numa2", "disk", "mixed", "compressed"]:
        print(f"'{s}' -> {DeviceType.convert(s)}")
    
    # 测试 NUMA 检测
    print("\n测试 NUMA 检测：")
    print(f"可用的 NUMA 设备: {DeviceType.get_available_numa_devices()}")
    
    # 测试 NUMA 节点 ID 转换
    print("\n测试 NUMA 节点 ID 转换：")
    print(f"NUMA0.to_numa_node_id() = {DeviceType.NUMA0.to_numa_node_id()}")
    print(f"NUMA1.to_numa_node_id() = {DeviceType.NUMA1.to_numa_node_id()}")
    print(f"NUMA2.to_numa_node_id() = {DeviceType.NUMA2.to_numa_node_id()}")
    
    # 测试设备类型检查
    print("\n测试设备类型检查：")
    print(f"NUMA0.is_numa_device() = {DeviceType.NUMA0.is_numa_device()}")
    print(f"DISK.is_numa_device() = {DeviceType.DISK.is_numa_device()}")

def test_torch_device():
    """测试 TorchDevice 类与 DeviceType 的交互"""
    print("\n测试 TorchDevice 创建：")
    
    # 测试不同的构造方式
    devices = [
        TorchDevice("numa0"),
        TorchDevice("cpu", numa_node=0),
        TorchDevice("disk"),
        TorchDevice("mixed"),
        TorchDevice("compressed")
    ]
    
    for i, dev in enumerate(devices):
        print(f"设备 {i}: {dev}, DeviceType = {dev.DeviceType}, numa_node = {dev.numa_node}")

def test_execution_env():
    """测试 ExecutionEnv 类与 DeviceType 的交互"""
    print("\n测试 ExecutionEnv：")
    
    # 创建执行环境
    offload_dir = "/tmp/flexllmgen_test"
    os.makedirs(offload_dir, exist_ok=True)
    
    # 获取可用的 NUMA 节点
    numa_nodes = list(range(torch_numa.get_numa_nodes()))
    print(f"系统中的 NUMA 节点: {numa_nodes}")
    
    # 创建执行环境
    env = ExecutionEnv.create_env(offload_dir, numa_nodes)
    
    # 测试 NUMA 设备获取
    print(f"可用的 NUMA 节点: {env.get_available_numa_nodes()}")
    
    # 测试每个设备
    for node_id in env.get_available_numa_nodes():
        device = env.get_numa_device(node_id)
        print(f"NUMA{node_id} 设备: {device}, DeviceType = {device.DeviceType}")
    
    # 测试其他设备
    print(f"磁盘设备: {env.disk}, DeviceType = {env.disk.DeviceType}")
    print(f"混合设备: {env.mixed}, DeviceType = {env.mixed.DeviceType}")

def test_policy_device_config():
    """测试 Policy 类与设备配置的交互"""
    print("\n测试 Policy 设备配置：")
    
    # 创建测试设备配置
    device_config = {
        'weight': {'numa0': 50, 'numa1': 30, 'disk': 20},
        'cache': {'numa0': 70, 'numa1': 30},
        'activation': {'numa0': 100}
    }
    
    # 创建策略
    policy = Policy.create_from_device_config(
        batch_size=4,
        num_batches=1,
        device_config=device_config,
        overlap=True,
        sep_layer=True,
        attn_sparsity=1.0,
        comp_weight=False,
        comp_weight_config=None,
        comp_cache=False,
        comp_cache_config=None
    )
    
    # 测试设备百分比获取
    for component in ['weight', 'cache', 'activation']:
        print(f"\n组件: {component}")
        for device in ['numa0', 'numa1', 'numa2', 'disk']:
            percent = policy.get_device_percent(component, device)
            print(f"  {device}: {percent}%")

if __name__ == "__main__":
    # 运行所有测试
    test_device_type_enum()
    test_torch_device()
    test_execution_env()
    test_policy_device_config()
