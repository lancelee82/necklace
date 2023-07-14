"""Model Wrapper for backward with activations checkpointing (for ZeRO)"""
#
#
#
# TODO:
#   1. ZeRO use activations checkpointing with pt-function style
#      Training Deep Nets with Sublinear Memory Cost
#      <https://github.com/Lyken17/pytorch-memonger>
#
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from necklace.frmwrk.pytorch.mpnn import checkpointing as pt_ckpt
from necklace.frmwrk.pytorch.pttsutils import pt_is_tensor


# ------------------------------------------------------------------
# show more info when backward error
# ------------------------------------------------------------------
#torch.autograd.set_detect_anomaly(True)


# ================================================================
# TODO: use checkpointing with pt-function style
# ================================================================
class MdWrpWithCkpt(nn.Module):
    def __init__(self, submdls=[], md_device='cuda:0', mp_order=0):
        super(MdWrpWithCkpt, self).__init__()

        self.submdls = submdls
        self.md_device = md_device
        self.mp_order = mp_order

        #submdls = [m.to(md_device) for m in submdls]
        net = nn.Sequential(*submdls)  # TODO: nn.ModuleList
        self.net = net#.to(md_device)

    def forward(self, *args, **kwargs):
        out = pt_ckpt.checkpoint(self.net.forward, *args)
        return out


# NOTE: now we can NOT put functions like torch.xx and F.xx to nn.Sequential,
#       and the forward of the submodel may be very special or complex,
#       so in the most cases we must implement submodel mannually with
#       a whole forward function, and then wrap them with this class
class MdWrpWithSelfBkwd(nn.Module):
    def __init__(self, submdls=[], md_device='cuda:0', mp_order=0,
                 is_last_part=False, do_acts_ckpt=True):
        super(MdWrpWithSelfBkwd, self).__init__()

        self.submdls = submdls
        self.md_device = md_device
        self.mp_order = mp_order
        self.is_last_part = is_last_part
        self.do_acts_ckpt = do_acts_ckpt  # TODO:

        #submdls = [m.to(md_device) for m in submdls]
        net = nn.Sequential(*submdls)  # TODO: nn.ModuleList
        self.net = net#.to(md_device)

        self.inp_cache = None
        self.out_cache = None

    def cache_graded_input(self, *args, **kwargs):
        if self.mp_order > 0:  # not the first submodule which is not need to get output gradients
            inputs = []
            for a in args:
                if pt_is_tensor(a):
                    # NOTE: here the `a` may be a Tensor with requires_grad=True and so it is not a
                    # leaf of graph and we can NOT save its grads, so here we must detch it from the graph
                    a = a.detach()
                    a.requires_grad = True
                inputs.append(a)
        else:
            inputs = args

        self.inp_cache = inputs

        return inputs

    def forward(self, *args, **kwargs):
        inputs = self.cache_graded_input(*args)

        # ================================================================================
        # TODO: (do_acts_ckpt)
        #       if True (for now), we will do the forward without grad, and do the forward
        #       again with grad in the my_backward()
        # ================================================================================
        if self.is_last_part or (not self.do_acts_ckpt):
            with torch.enable_grad():
                out = self.net(*inputs)

        else:
            with torch.no_grad():
                out = self.net(*inputs)

        self.out_cache = out  # NOTE: cache the out for mp_backward

        return out

    def get_input_grads(self):
        if self.mp_order > 0:
            grads = [inp.grad for inp in self.inp_cache]
        else:
            grads = None

        return grads

    def mp_backward(self, grads=None, *args, **kwargs):

        # ================================================================================
        # TODO: (do_acts_ckpt)
        #       if True (for now), we will do the forward without grad, and do the forward
        #       again with grad in the my_backward()
        # SO, it is very slow !!!
        # and, this is NOT the real activations_checkpoint, see `pt_ckpt` for details
        # ================================================================================

        if grads is not None:  # not the last submodule which has already done the backward with loss

            #if not self.is_last_part:
            if self.do_acts_ckpt:
                with torch.enable_grad():
                    out = self.net(*self.inp_cache)
                self.out_cache = out
            else:
                pass

            if not isinstance(self.out_cache, (list, tuple, set)):
                outs = [self.out_cache]
            else:
                outs = self.out_cache

            for out, grad in zip(outs, grads):
                torch.autograd.backward(out, grad)
                # or,
                #out.backward(grad)

        inp_grads = self.get_input_grads()

        return inp_grads


class MdWrpWholeNet(nn.Module):
    def __init__(self, mdls=[], md_device='cpu'):
        super(MdWrpWholeNet, self).__init__()
        self.whole_md = nn.Sequential(*mdls)

    def forward(self, *args, **kwargs):
        out = self.whole_md(*args)
        return out
