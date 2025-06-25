# FlexGen_NUMA
FlexGen with NUMA-aware memory allocation and simplified device configuration.

## Basic Usage

### Environment Setup
```bash
export CUDA_VISIBLE_DEVICES=""  # Disable GPU
```

### Simple Commands
```bash
# Use default HuggingFace mirror (hf-mirror.com)
python -m flexgen.main --model facebook/opt-125m --path __DUMMY__ --weight_devices numa0:100 --cache_devices numa0:100 --activation_devices numa0:100

# Specify custom mirror
python -m flexgen.main --model facebook/opt-125m --path __DUMMY__ --hf_mirror https://hf-mirror.com --weight_devices numa0:100 --cache_devices numa0:100 --activation_devices numa0:100

# Mixed NUMA and disk allocation
python -m flexgen.main --model facebook/opt-125m --path __DUMMY__ --weight_devices numa0:70,disk:30 --cache_devices numa0:80,disk:20 --activation_devices numa0:100
```

### Device Configuration
- `--weight_devices`: Weight tensor placement (e.g., `numa0:100` or `numa0:70,disk:30`)
- `--cache_devices`: Cache tensor placement 
- `--activation_devices`: Activation tensor placement
- `--hf_mirror`: HuggingFace mirror URL (default: https://hf-mirror.com)

