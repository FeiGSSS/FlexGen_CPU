export CUDA_VISIBLE_DEVICES=""
export HF_ENDPOINT=https://hf-mirror.com


# No Offload, all in CPU
weight_cpu=100
cache_cpu=100
act_cpu=100

weight_numa=0
cache_numa=0
act_numa=0

echo "Running FlexGen with CPU Offload"

python -m flexgen.main --model facebook/opt-125m --path __DUMMY__ \
       --percent $weight_cpu $cache_cpu $act_cpu $weight_numa $cache_numa $act_numa

# Offload weight and cache to numa
weight_cpu=0
cache_cpu=0
act_cpu=100

weight_numa=100
cache_numa=100
act_numa=0

echo "Running FlexGen with NUMA Offload"
python -m flexgen.main --model facebook/opt-125m --path __DUMMY__ \
       --percent $weight_cpu $cache_cpu $act_cpu $weight_numa $cache_numa $act_numa

# Offload weight and cache to Disk
weight_cpu=0
cache_cpu=0
act_cpu=100

weight_numa=0
cache_numa=0
act_numa=0

echo "Running FlexGen with Disk Offload"
python -m flexgen.main --model facebook/opt-125m --path __DUMMY__ \
       --percent $weight_cpu $cache_cpu $act_cpu $weight_numa $cache_numa $act_numa