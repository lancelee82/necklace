"""MP (Model-Parallel) utils"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelPartWrapper(nn.Module):
    def __init__(self, submdls=[], md_device='cuda:0', mp_order=0):
        super(ModelPartWrapper, self).__init__()

        self.submdls = submdls
        self.md_device = md_device
        self.mp_order = mp_order

        submdls = [m.to(md_device) for m in submdls]
        net = nn.Sequential(*submdls)  # TODO: nn.ModuleList
        self.net = net.to(md_device)

        self.inp_cache = None
        self.out_cache = None

    def cache_graded_input(self, *args, **kwargs):
        if self.mp_order > 0:  # not the first submodule which is not need to get output gradients
            inputs = []
            for a in args:
                # NOTE: here the `a` may be a Tensor with requires_grad=True and so it is not a
                # leaf of graph and we can NOT save its grads, so here we must detch it from the graph
                a = a.detach()
                a.requires_grad = True
                inputs.append(a)

            self.inp_cache = inputs
        else:
            inputs = args
        return inputs

    # NOTE: now we can NOT put functions like torch.xx and F.xx to nn.Sequential,
    #       and the forward of the submodel may be very special or complex,
    #       so in the most cases we must implement submodel mannually with
    #       a whole forward function, and then wrap them with this class
    def forward(self, *args, **kwargs):
        inputs = self.cache_graded_input(*args)
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
        if grads is not None:  # not the last submodule which has already done the backward with loss
            if not isinstance(self.out_cache, (list, tuple, set)):
                outs = [self.out_cache]
            for out, grad in zip(outs, grads):
                torch.autograd.backward(out, grad)
                # or,
                #out.backward(grad)

        grads = self.get_input_grads()

        return grads


class MpTrnrModulesMap(object):
    pass  # TODO:
