export CUDA_VISIBLE_DEVICES=""
export HF_ENDPOINT=https://hf-mirror.com

model=facebook/opt-2.7b

act_cpu=100
act_numa=0

# No Offload, all in CPU
weight_cpu=100
cache_cpu=100

weight_numa=0
cache_numa=0

echo "Running FlexGen with CPU Offload"

numactl --cpunodebind=0,1 --membind=0,1 python -m flexgen.main --model $model --path __DUMMY__ \
       --percent $weight_cpu $cache_cpu $act_cpu $weight_numa $cache_numa $act_numa

# Offload weight and cache to numa
weight_cpu=0
cache_cpu=0

weight_numa=100
cache_numa=100

echo "Running FlexGen with NUMA Offload"
numactl --cpunodebind=0,1 --membind=0,1 python -m flexgen.main --model $model --path __DUMMY__ \
       --percent $weight_cpu $cache_cpu $act_cpu $weight_numa $cache_numa $act_numa

# Offload weight and cache to Disk
weight_cpu=0
cache_cpu=0

weight_numa=0
cache_numa=0

echo "Running FlexGen with Disk Offload"
numactl --cpunodebind=0,1 --membind=0,1 python -m flexgen.main --model $model --path __DUMMY__ \
       --percent $weight_cpu $cache_cpu $act_cpu $weight_numa $cache_numa $act_numa