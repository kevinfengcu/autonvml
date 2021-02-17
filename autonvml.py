import os
import warnings
from functools import total_ordering

import pynvml as nvml


@total_ordering
class GPU:
    def __init__(self, idx):
        self.idx = idx
        self.handler = nvml.nvmlDeviceGetHandleByIndex(idx)
        self.available = True
        self.query()
    def query(self):
        h = self.handler
        self.long_name = nvml.nvmlDeviceGetName(h).decode('utf-8')
        self.short_name = self.long_name.split(' ')[-1]
        if self.short_name[:-2] != 'Ti':
            self.short_name = int(self.short_name)
        else:
            self.short_name = int(self.short_name[:-2]) + 0.5
        self.totalmem = nvml.nvmlDeviceGetMemoryInfo(h).total/1000000
        self.freemem = nvml.nvmlDeviceGetMemoryInfo(h).free/1000000
        self.utilrate = nvml.nvmlDeviceGetUtilizationRates(h).gpu
    def __lt__(self,other):
        return (self.short_name < other.short_name) or (self.short_name == other.short_name and self.utilrate > other.utilrate)
    def __eq__(self,other):
        return self.short_name == other.short_name

def enum_gpus(idx_list = None):
    if idx_list == None:
        num_gpus = nvml.nvmlDeviceGetCount()
        idx_list = range(num_gpus)
    return [GPU(i) for i in idx_list]

def filter_gpus(gpu_list, utilrate = 50, freemem = 8000):
    avail_gpu_list = []
    for i in gpu_list:
        i.query()
        if i.utilrate <= utilrate and i.freemem >= freemem:
            avail_gpu_list.append(i)
    return avail_gpu_list

def set_cuda_gpu_env(gpu_list = []):
    if len(gpu_list) == 0:
        s = ''
    else:
        s = ','.join(str(i.idx) for i in gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = s

def grab_gpus(num = 1, utilrate = 50, freemem = 8000, set_cuda = True, idx_list = None):
    gpu_list = enum_gpus(idx_list)
    avail_gpu_list = filter_gpus(gpu_list,utilrate,freemem)
    sorted_gpu_list = sorted(avail_gpu_list,reverse=True)[:num]
    if len(sorted_gpu_list) < num:
        warnings.warn(f"Only {len(sorted_gpu_list)} GPU(s) is/are available.", RuntimeWarning)
    if set_cuda:
        set_cuda_gpu_env(sorted_gpu_list)
    return sorted_gpu_list

nvml.nvmlInit()

