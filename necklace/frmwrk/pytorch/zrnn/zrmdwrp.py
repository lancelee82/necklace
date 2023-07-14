"""ZeRO Module / Model Wrapper"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zrfn import bcast_to_zero_parallel_region


class ZrModuleWrapper(nn.Module):
    def __init__(self, md, zr_rank, cfg=None, *args, **kwargs):
        super(ZrModuleWrapper, self).__init__()

        self.md = md
        self.zr_rank = zr_rank
        self.cfg = cfg


    def forward(self, x):
        # md to GPU
        # get params flatten memory (prms_flt_mem)
        # bcast_to_zero_parallel_region(prms_flt_mem)
        # re-construct md
        # o = md.forward(x)
        # md to CPU  # TODO: ???
        pass  # TODO:
        # TODO: activations checkpointing [================================]
