/**
 * @file torch_numa.cpp
 * @brief PyTorch NUMA (Non-Uniform Memory Access) 内存分配器扩展
 * 
 * 本文件实现了一个 NUMA 感知的内存分配器，允许在指定的 NUMA 节点上分配内存，
 * 从而优化多 NUMA 节点系统中的内存访问性能。
 * 
 * 主要功能：
 * - NUMA 节点特定的内存分配
 * - 张量在 NUMA 节点间的迁移
 * - 线程到 NUMA 节点的绑定
 * - 默认 NUMA 节点设置
 * 
 * @author PyTorch NUMA Extension
 * @date 2024
 */

#define _GNU_SOURCE  // 启用 GNU 扩展功能
#include <torch/extension.h>
#include <numa.h>           // NUMA 库函数
#include <sched.h>          // CPU 调度相关函数
#include <sstream>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>

/**
 * @struct AllocationInfo
 * @brief 存储内存分配信息的结构体
 * 
 * 用于跟踪每个内存分配的大小和所在的 NUMA 节点，
 * 确保在释放内存时能正确调用 numa_free。
 */
struct AllocationInfo {
  size_t size;  ///< 分配的内存大小（字节）
  int node;     ///< 分配内存所在的 NUMA 节点
};

/// 全局映射表，记录所有通过 NUMA 分配器分配的内存信息
static std::unordered_map<void*, AllocationInfo> allocated_info;

/// 保护 allocated_info 的互斥锁，确保线程安全
static std::mutex allocation_mutex;

/**
 * @class NumaAllocator
 * @brief NUMA 感知的内存分配器类
 * 
 * 继承自 PyTorch 的 c10::Allocator 基类，实现在指定 NUMA 节点上分配内存。
 * 这个分配器确保分配的内存物理上位于指定的 NUMA 节点，从而提高
 * 本地内存访问性能。
 */
class NumaAllocator final : public c10::Allocator {
 private:
  int node;  ///< 目标 NUMA 节点 ID

 public:
  /**
   * @brief 构造函数
   * @param node 目标 NUMA 节点 ID
   */
  NumaAllocator(int node) : node(node) {}

  /**
   * @brief 在指定 NUMA 节点上分配内存
   * @param nbytes 要分配的字节数
   * @return c10::DataPtr 包含分配内存指针和删除器的智能指针
   * 
   * 使用 numa_alloc_onnode() 在指定节点分配内存，并记录分配信息
   * 以便后续正确释放。
   */
  c10::DataPtr allocate(size_t nbytes) override {
    // 处理零字节分配的边界情况
    if (nbytes == 0) {
      return c10::DataPtr(nullptr, c10::Device(c10::kCPU));
    }

    // 在指定 NUMA 节点分配内存
    void* raw = numa_alloc_onnode(nbytes, node);
    std::ostringstream oss;
    oss << "numa_alloc_onnode failed (node=" << node << ")";
    TORCH_CHECK(raw, oss.str());

    // 记录分配信息，用于后续释放
    {
      std::lock_guard<std::mutex> lock(allocation_mutex);
      allocated_info[raw] = {nbytes, node};
    }

    // 返回包含自定义删除器的 DataPtr
    return c10::DataPtr(
        raw,                              // 数据指针
        raw,                              // 上下文指针（用于删除器）
        &NumaAllocator::raw_deallocate,   // 删除器函数
        c10::Device(c10::kCPU));          // 设备类型
  }

  /**
   * @brief 静态内存释放函数
   * @param ptr 要释放的内存指针
   * 
   * 根据记录的分配信息，使用正确的大小调用 numa_free() 释放内存。
   */
  static void raw_deallocate(void* ptr) {
    if (!ptr) return;

    // 查找并移除分配信息
    AllocationInfo info = {0, 0};
    {
      std::lock_guard<std::mutex> lock(allocation_mutex);
      auto it = allocated_info.find(ptr);
      if (it != allocated_info.end()) {
        info = it->second;
        allocated_info.erase(it);
      }
    }

    // 使用正确的大小释放 NUMA 内存
    if (info.size > 0) {
      numa_free(ptr, info.size);
    }
  }

  /**
   * @brief 获取删除器函数指针
   * @return 删除器函数指针
   */
  c10::DeleterFnPtr raw_deleter() const override {
    return &NumaAllocator::raw_deallocate;
  }

  /**
   * @brief 数据拷贝函数
   * @param dest 目标内存地址
   * @param src 源内存地址
   * @param count 拷贝字节数
   * 
   * 使用标准 memcpy 进行数据拷贝。
   */
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

  /**
   * @brief 获取分配器对应的 NUMA 节点
   * @return NUMA 节点 ID
   */
  int get_node() const { return node; }
};

/**
 * @brief 获取张量所在的 NUMA 节点
 * @param tensor 要查询的张量
 * @return int NUMA 节点 ID，如果无法确定则返回 -1
 * 
 * 通过查询内部分配信息表来确定张量的内存位于哪个 NUMA 节点。
 * 只有通过我们的 NUMA 分配器分配的张量才能被正确识别。
 */
int get_tensor_numa_node(const torch::Tensor& tensor) {
  void* ptr = tensor.data_ptr();
  
  // 在分配信息表中查找
  std::lock_guard<std::mutex> lock(allocation_mutex);
  auto it = allocated_info.find(ptr);
  if (it != allocated_info.end()) {
    return it->second.node;
  }
  
  // 如果不是通过我们的分配器分配的，返回未知
  return -1;
}

/**
 * @brief 在指定 NUMA 节点上创建空张量（类似 torch.empty）
 * @param sizes 张量的维度大小列表
 * @param node 目标 NUMA 节点 ID
 * @param dtype 张量的数据类型，默认为 float32
 * @return torch::Tensor 在指定节点创建的空张量
 * 
 * 这个函数设计为直接替换 torch.empty，提供相同的接口但支持 NUMA 节点指定。
 */
torch::Tensor _tensor_on_node(std::vector<int64_t> sizes, int node, c10::ScalarType dtype) {
  // 验证节点 ID 的有效性
  int max_node = numa_max_node();
  TORCH_CHECK(node >= 0 && node <= max_node, "Invalid NUMA node: ", node, " (max: ", max_node, ")");
  
  // 创建或复用指定节点的分配器
  // 注意：这里使用静态指针是为了避免频繁创建分配器对象
  static NumaAllocator* current_allocator = nullptr;
  if (!current_allocator || current_allocator->get_node() != node) {
    current_allocator = new NumaAllocator(node);
    c10::SetAllocator(c10::kCPU, current_allocator);
  }
  
  // 设置张量选项
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  
  // 创建空张量，会自动使用当前设置的 NUMA 分配器
  auto tensor = torch::empty(sizes, options);
  
  return tensor;
}

/**
 * @brief 将张量迁移到指定 NUMA 节点（类似 tensor.to()）
 * @param tensor 源张量
 * @param node 目标 NUMA 节点 ID
 * @return torch::Tensor 在目标节点的新张量
 * 
 * 仿照 PyTorch 的 tensor.to() 方法，将张量迁移到指定的 NUMA 节点。
 * 如果张量已经在目标节点，直接返回原张量。
 */
torch::Tensor to(const torch::Tensor& tensor, int node) {
  // 检查是否已经在目标节点
  int current_node = get_tensor_numa_node(tensor);
  if (current_node == node) {
    return tensor;  // 已经在目标节点，直接返回
  }
  
  // 在目标节点创建相同形状和类型的新张量
  auto new_tensor = _tensor_on_node(tensor.sizes().vec(), node, tensor.scalar_type());
  
  // 拷贝数据
  new_tensor.copy_(tensor);
  
  return new_tensor;
}

/**
 * @brief 将当前线程绑定到指定的 NUMA 节点
 * @param node 目标 NUMA 节点 ID
 * 
 * 使用 NUMA 库的 numa_bind() 函数将当前线程绑定到指定节点。
 * 绑定后，线程倾向于在该节点的 CPU 上运行，提高内存访问局部性。
 */
void bind_to_numa_node(int node) {
  struct bitmask* mask = numa_allocate_nodemask();
  numa_bitmask_setbit(mask, node);
  numa_bind(mask);
  numa_free_nodemask(mask);
}

/**
 * @brief 获取当前线程所在的 NUMA 节点
 * @return int 当前 NUMA 节点 ID，失败时返回 -1
 * 
 * 通过获取当前 CPU ID 并查询其所属的 NUMA 节点来确定线程位置。
 */
int get_current_numa_node() {
  int cpu = sched_getcpu();
  if (cpu >= 0) {
    return numa_node_of_cpu(cpu);
  }
  return -1;
}

/**
 * @brief 注册默认的 NUMA 分配器（节点 0）
 * 
 * 将全局的 CPU 分配器替换为 NUMA 分配器，默认使用节点 0。
 * 调用此函数后，所有新创建的 CPU 张量都将使用 NUMA 分配器。
 */
void register_numa_allocator() {
  static NumaAllocator alloc(0);
  c10::SetAllocator(c10::kCPU, &alloc);
}

/**
 * @brief 获取系统可用的 NUMA 节点数量
 * @return int NUMA 节点数量，如果 NUMA 不可用则返回 0
 */
int get_numa_nodes() {
  if (numa_available() < 0) {
    return 0;
  }
  return numa_num_configured_nodes();
}

/**
 * @brief 设置默认的 NUMA 节点
 * @param node 要设置为默认的 NUMA 节点 ID
 * 
 * 将全局 CPU 分配器替换为指定节点的 NUMA 分配器。
 * 之后创建的所有 CPU 张量都将分配在指定节点上。
 */
void set_default_numa_node(int node) {
  static NumaAllocator alloc(node);
  c10::SetAllocator(c10::kCPU, &alloc);
}

/**
 * @brief Python 模块绑定
 * 
 * 使用 pybind11 将 C++ 函数暴露给 Python。
 * 定义了模块的所有公共接口函数。
 */
PYBIND11_MODULE(torch_numa_cpp, m) {  // 修改模块名
  m.doc() = "PyTorch NUMA memory allocator extension";
  
  // 分配器管理函数
  m.def("register_numa_allocator", &register_numa_allocator, 
        "Register default NUMA allocator (node 0)");
  m.def("set_default_numa_node", &set_default_numa_node, 
        "Set default NUMA node for all CPU tensor allocation",
        py::arg("node"));
  
  // 信息查询函数
  m.def("get_numa_nodes", &get_numa_nodes, 
        "Get number of available NUMA nodes");
  m.def("get_current_numa_node", &get_current_numa_node, 
        "Get current thread's NUMA node");
  
  // 张量操作函数 - 修复默认参数
  m.def("_tensor_on_node", &_tensor_on_node, 
        "Create empty tensor on specific NUMA node (replacement for torch.empty)", 
        py::arg("sizes"), py::arg("node"), py::arg("dtype") = c10::kFloat);
  m.def("to", &to, 
        "Move tensor to specific NUMA node (like tensor.to())",
        py::arg("tensor"), py::arg("node"));
  
  // 保留向后兼容的接口
  m.def("create_tensor_on_node", &_tensor_on_node, 
        "Create tensor on specific NUMA node (deprecated, use _tensor_on_node)", 
        py::arg("sizes"), py::arg("node"), py::arg("dtype") = c10::kFloat);
  m.def("migrate_tensor_to_node", &to, 
        "Migrate tensor to specific NUMA node (deprecated, use to)",
        py::arg("tensor"), py::arg("node"));
  
  m.def("get_tensor_numa_node", &get_tensor_numa_node, 
        "Get NUMA node of tensor");
  
  // 线程管理函数
  m.def("bind_to_numa_node", &bind_to_numa_node, 
        "Bind current thread to specific NUMA node",
        py::arg("node"));
}