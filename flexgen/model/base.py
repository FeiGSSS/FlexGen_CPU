import numpy as np
from typing import Optional, List

from flexgen.utils import (
    Task,
    ExecutionEnv,
    Policy,
    DUMMY_WEIGHT,
    torch_dtype_to_np_dtype
)

def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]

class BaseModel:
    def __init__(self):
        self.task: Optional[Task] = None
    
    def set_task(self, task: Task):
        self.task = task
        
    def init_weight(self, *args):
        raise NotImplementedError()
    
    def load_weight(self, *args):
        raise NotImplementedError()
    
    def init_cache_one_batch(self, *args):
        pass
    
    def load_cache(self, *args):
        pass
    
    def store_cache(self, *args):
        pass
    
    def forward(self, *args):
        raise NotImplementedError()
    
    def init_weight_list(self,
                         weight_specs: List[tuple],
                         policy: Policy,
                         env: ExecutionEnv):
        # Multi-NUMA and disk device configuration
        dev_percents = []
        dev_choices = []
        
        # Add disk device if configured
        w_disk_percent = policy.get_device_percent('weight', 'disk')
        if w_disk_percent > 0:
            dev_percents.append(w_disk_percent)
            dev_choices.append(env.disk)
        
        # Add all configured NUMA devices
        available_numa_nodes = env.get_available_numa_nodes()
        for numa_id in available_numa_nodes:
            numa_percent = policy.get_device_percent('weight', f'numa{numa_id}')
            if numa_percent > 0:
                dev_percents.append(numa_percent)
                dev_choices.append(env.get_numa_device(numa_id))
        
        # Default to NUMA0 if no configuration
        if not dev_choices:
            dev_percents = [100]
            dev_choices = [env.get_numa_device(0)]

        sizes = [np.prod(spec[0]) for spec in weight_specs]
        sizes_cumsum = np.cumsum(sizes)
        ret = []
        for i in range(len(weight_specs)):
            mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
            home = get_choice(mid_percent * 100, dev_percents, dev_choices)
            shape, dtype, filename = weight_specs[i]
            
            if len(shape) < 2:
                compress = False
            else:
                compress = policy.comp_weight
                
            if not compress:
                weight = home.allocate(shape, dtype)

                if DUMMY_WEIGHT not in filename:
                    weight.load_from_np_file(filename)
                else:
                    weight.load_from_np(np.ones(shape, dtype))
            else:
                weight = home.compressed_device.allocate(shape, dtype, policy.comp_weight_config)

                if DUMMY_WEIGHT not in filename:
                    weight.load_from_np_file(filename)
                else:
                    for i in range(2):
                        x = weight.data[i]
                        x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

            ret.append(weight)
        return ret
    
    