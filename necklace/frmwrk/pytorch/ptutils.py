"""Pytorch Utils"""

import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def pt_dtype_to_str(dtype):
    return str(dtype)[6:]  # 'torch.float32' --> 'float32'


def clip_grad_norm(model, clip=0.25, *args, **kwargs):
    # `clip_grad_norm` helps prevent the exploding gradient
    # problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


def get_state_dict_from_dp(pth_tar, state_key='state_dict'):
    # original saved file with DataParallel
    checkpoint = torch.load(pth_tar)
    state_dict = checkpoint[state_key]

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    # load params
    #model.load_state_dict(new_state_dict)
    
    return new_state_dict


# ----------------------------------------------------------------------------
"""Model Utils"""

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


def prnt_md(model):
    print(model)
    num_params = 0
    lines = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        #print(name, '\t\t', p.shape, '\t\t', p.numel())
        lines.append([name, str(p.shape), str(p.numel())])
        num_params += p.numel()
    prnt_by_tab(lines)
    print(" | > Model has {} parameters".format(num_params))


def prnt_by_tab(a):
    n = len(a[0])
    mxs = []
    for i in range(n):
        mx = max([len(aa[i]) for aa in a])
        mxs.append(mx)

    for j in range(len(a)):
        line = pad_line(a[j], mxs)
        print('\t'.join(line))


def pad_line(aj, mxs):
    a = []
    for i in range(len(aj)):
        a.append(aj[i] + (' ' * (mxs[i] - len(aj[i]))))
    return a


# ----------------------------------------------------------------------------
# cpu gpu memory utils

import gc

def clear_tensor_mem(t=None, do_gc=False, do_pt_cache=True):
    if t is not None:
        del t

    if do_gc:
        gc.collect()

    if do_pt_cache:
        torch.cuda.empty_cache()


def see_memory_usage(message, force=False):
    if not force:
        return
    print(message)
    print("Memory Allocated ", torch.cuda.memory_allocated()/(1024*1024*1024), "GigaBytes")
    print("Max Memory Allocated ", torch.cuda.max_memory_allocated()/(1024*1024*1024), "GigaBytes")
    print("Cache Allocated ", torch.cuda.memory_reserved()/(1024*1024*1024), "GigaBytes")
    print("Max cache Allocated ", torch.cuda.max_memory_reserved()/(1024*1024*1024), "GigaBytes")
    print(" ")


def show_cuda_memory_usage(message, force=False):
    if not force:
        return

    print(message)
    print("Memory Allocated     ", torch.cuda.memory_allocated())
    print("Max Memory Allocated ", torch.cuda.max_memory_allocated())
    print("Cache Allocated      ", torch.cuda.memory_reserved())
    print("Max cache Allocated  ", torch.cuda.max_memory_reserved())
    print(" ")


class PtCudaMemUsage(object):
    def __init__(self, by_unit=1, do_print=False, clear_mem=False):
        self.by_unit = by_unit
        self.do_print = do_print
        self.clear_mem = clear_mem

        self.memory_allocated_old = 0
        self.max_memory_allocated_old = 0
        self.memory_reserved_old = 0
        self.max_memory_reserved_old = 0

        self.memory_allocated_now = 0
        self.max_memory_allocated_now = 0
        self.memory_reserved_now = 0
        self.max_memory_reserved_now = 0

    def show(self, msg):
        if not self.do_print:
            return

        if self.clear_mem:
            clear_tensor_mem(do_gc=True, do_pt_cache=True)

        print(msg)

        self.memory_allocated_old = self.memory_allocated_now
        self.max_memory_allocated_old = self.max_memory_allocated_now
        self.memory_reserved_old = self.memory_reserved_now
        self.max_memory_reserved_old = self.max_memory_reserved_now

        self.memory_allocated_now = torch.cuda.memory_allocated() // self.by_unit
        self.max_memory_allocated_now = torch.cuda.max_memory_allocated() // self.by_unit
        self.memory_reserved_now = torch.cuda.memory_reserved() // self.by_unit
        self.max_memory_reserved_now = torch.cuda.max_memory_reserved() // self.by_unit

        print("Memory Allocated     ", self.memory_allocated_now)
        print("Max Memory Allocated ", self.max_memory_allocated_now)
        print("Cache Allocated      ", self.memory_reserved_now)
        print("Max cache Allocated  ", self.max_memory_reserved_now)

        print("Memory Allocated     ::", self.memory_allocated_now - self.memory_allocated_old)
        print("Max Memory Allocated ::", self.max_memory_allocated_now - self.max_memory_allocated_old)
        print("Cache Allocated      ::", self.memory_reserved_now - self.memory_reserved_old)
        print("Max cache Allocated  ::", self.max_memory_reserved_now - self.max_memory_reserved_old)




# ----------------------------------------------------------------------------
# /microsoft_DeepSpeed/deepspeed/runtime/utils.py
mem_alloced = 0
mem_cached = 0


def memory_status(msg, print_rank=-1, reset_max=False):
    global mem_alloced, mem_cached

    rank = 0#dist.get_rank()
    if print_rank != -1 and rank != print_rank:
        return

    torch.cuda.synchronize()

    if reset_max:
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()

    new_alloced = torch.cuda.memory_allocated()
    new_cached = torch.cuda.memory_cached()

    delta_alloced = new_alloced - mem_alloced
    delta_cached = new_cached - mem_cached

    mem_cached = new_cached
    mem_alloced = new_alloced

    max_alloced = torch.cuda.max_memory_allocated()
    max_cached = torch.cuda.max_memory_cached()

    # convert to GB for printing
    new_alloced /= 1024**3
    new_cached /= 1024**3
    delta_alloced /= 1024**3
    delta_cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f'RANK={rank} MEMSTATS',
        msg,
        f'device={torch.cuda.current_device()} '
        f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
        f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)'
    )


def get_ma_status():
    #if dist.is_initialized() and not dist.get_rank() == 0:
    #    return 0
    return torch.cuda.memory_allocated()
