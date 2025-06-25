# PyTorch NUMA Extension

🚀 **NUMA-aware memory allocator for PyTorch - boost performance on multi-NUMA systems**

## Quick Start

```bash
# Install dependencies
sudo apt-get install libnuma-dev numactl

# Build extension
python setup.py build_ext --inplace

# Test installation
python -c "import torch_numa; print(f'NUMA nodes: {torch_numa.get_numa_nodes()}')"
```

## Usage

```python
import torch_numa

# Create tensors on specific NUMA nodes
tensor = torch_numa.empty(100, 100, node=0)        # PyTorch style args
zeros = torch_numa.zeros([50, 50], node=1)         # or list style
randn = torch_numa.randn(200, 200, node=0, dtype=torch.float64)

# Check and move tensors
print(f"Tensor on node: {tensor.numa_node}")
migrated = tensor.numa_to(1)  # Move to node 1

# Bind threads and set defaults
torch_numa.bind_to_numa_node(0)
torch_numa.set_default_numa_node(1)
```

## Neural Network Example

```python
import torch.nn as nn

class NumaModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch_numa.set_default_numa_node(0)
        self.layer1 = nn.Linear(784, 256)
        
        torch_numa.set_default_numa_node(1) 
        self.layer2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = x.numa_to(1)  # Move to node 1
        return self.layer2(x)

# Usage
model = NumaModel()
data = torch_numa.randn(32, 784, node=0)
output = model(data)
```

## API Reference

### Tensor Creation
- `torch_numa.empty(*size, node=0, **kwargs)` - Empty tensor
- `torch_numa.zeros(*size, node=0, **kwargs)` - Zero tensor  
- `torch_numa.ones(*size, node=0, **kwargs)` - Ones tensor
- `torch_numa.randn(*size, node=0, **kwargs)` - Random normal
- `torch_numa.rand(*size, node=0, **kwargs)` - Random uniform

### Tensor Operations
- `tensor.numa_node` - Get tensor's NUMA node
- `tensor.numa_to(node)` - Move tensor to node

### System Control
- `torch_numa.get_numa_nodes()` - Number of NUMA nodes
- `torch_numa.bind_to_numa_node(node)` - Bind thread to node
- `torch_numa.set_default_numa_node(node)` - Set default node


## Requirements

- Linux with NUMA support (2+ nodes recommended)
- Python 3.7+, PyTorch 2.0+
- libnuma-dev, numactl