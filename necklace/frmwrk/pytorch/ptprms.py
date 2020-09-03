""""""
import copy
from collections import OrderedDict

import numpy as np

import torch

from necklace.cuda import nbutils


# -------------------------------------------------------------------------
# from \snipe\snipe\mdwrp_pt.py

def get_weights(model):
    state_dict = model.state_dict()

    # convert to numpy for msgpck
    for k, v in state_dict.items():
        state_dict[k] = state_dict[k].cpu().numpy()

    return state_dict


def set_weights(model, weights):
    # convert from numpy for msgpck
    for k, v in weights.items():
        weights[k] = torch.from_numpy(v)#.float()

    model.load_state_dict(weights)


def collect_gradients_by_optimizer(optimizer):
    grads = []

    # \pytorch\torch\optim\optimizer.py  zero_grad()
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                # TODO: use dict for save param names
                # grads.append(p.grad)  # NOTE: use numpy for msgpck
                grads.append(p.grad.data.cpu().numpy())
            else:
                pass  # TODO: all 0.0 [?]

    return grads


def collect_gradients_by_parameters(model):
    grads = []

    for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.cpu().numpy())
            else:
                pass  # TODO: all 0.0 [?]

    return grads


def grads_get(model):
    grads = OrderedDict()

    for name, p in model.named_parameters():
        if p.grad is not None:  # or p.requires_grad
            grads[name] = p.grad#.data
        else:
            pass

    return grads


# -------------------------------------------------------------------------
# for do nccl

def init_params_recvs(model):
    prm_revs = OrderedDict()

    for name, p in model.named_parameters():

        if p.grad is None:  # or not p.requires_grad
            continue

        #if p.grad is not None:  # or p.requires_grad
        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:
            a = torch.zeros(shp)
            if p.is_cuda:
                a = a.to(p.device)
            prm_revs[name] = a

    return prm_revs


def params_do_nccl_allreduce(nc, kn, model, prm_revs=None):

    for name, p in model.named_parameters():

        if p.grad is None:  # or not p.requires_grad
            continue

        #if p.grad is not None:  # or p.requires_grad
        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = p.data
            #d_arr_recv = prm_revs[name]

            sz = np.prod(d_arr_send.size()) #* d_arr_send.element_size()
            nc.do_all_reduce(d_arr_send.data_ptr(),
                             d_arr_send.data_ptr(),
                             sz)
            nc.stream_sync()

            d_arr_send[:] = d_arr_send / float(kn)


def grads_do_nccl_allreduce__1(nc, kn, model, prm_revs=None, grad_comp_cfg=None):

    for name, p in model.named_parameters():

        if p.grad is None:  # or not p.requires_grad
            continue

        #if p.grad is not None:  # or p.requires_grad
        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = p.grad.data
            #d_arr_recv = prm_revs[name]

            sz = np.prod(d_arr_send.size()) #* d_arr_send.element_size()
            nc.do_all_reduce(d_arr_send.data_ptr(),
                             d_arr_send.data_ptr(),
                             sz)
            nc.stream_sync()

            d_arr_send[:] = d_arr_send / float(kn)


# -------------------------------------------------------------------------
# flatten tensors for do nccl with good performance

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors


MB = 1024 * 1024
# used for intra-node param sync and inter-node sync as well
broadcast_bucket_size = 10 * MB
nccl_reduce_bucket_size = 256 * MB


def nccl_allreduce_by_buckets(nc, kn, all_grads):
    # Now bucketing the parameters
    dev_grads_buckets = _take_tensors(all_grads, nccl_reduce_bucket_size)


    for grads_batch in dev_grads_buckets:
        grads_batch_coalesced = _flatten_dense_tensors(grads_batch)

        # NOTE:
        torch.cuda.synchronize()

        # NOTE:
        #nbutils.cuda_current_context().synchronize()
        # or,
        nc.stream_sync()

        sz = np.prod(grads_batch_coalesced.size())
        nc.do_all_reduce(grads_batch_coalesced.data_ptr(),
                         grads_batch_coalesced.data_ptr(),
                         sz)

        nc.stream_sync()

        grads_batch_coalesced[:] = grads_batch_coalesced / float(kn)

        grads_batch_reduced = _unflatten_dense_tensors(
            grads_batch_coalesced, grads_batch)

        for grad, reduced in zip(grads_batch, grads_batch_reduced):
            grad.copy_(reduced)


def grads_do_nccl_allreduce(nc, kn, model, prm_revs=None, grad_comp_cfg=None):

    all_grads = []

    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if param.grad.requires_grad:
            raise RuntimeError("DistributedDataParallel only works "
                               "with gradients that don't require "
                               "grad")
        # Adding the gradients for reduction
        all_grads.append(param.grad.data)

    nccl_allreduce_by_buckets(nc, kn, all_grads)


def opt_grads_do_nccl_allreduce(nc, kn, opt, prm_revs=None, grad_comp_cfg=None):
    if isinstance(opt, (list, tuple)):
        for o in opt:
            opt_one_grads_do_nccl_allreduce(
                nc, kn, o, prm_revs=prm_revs, grad_comp_cfg=grad_comp_cfg)
    else:
        opt_one_grads_do_nccl_allreduce(
            nc, kn, opt, prm_revs=prm_revs, grad_comp_cfg=grad_comp_cfg)


def opt_one_grads_do_nccl_allreduce(nc, kn, opt, prm_revs=None, grad_comp_cfg=None):

    all_grads = []

    for group in opt.param_groups:
        for p in group['params']:
            if p.grad is not None:
                all_grads.append(p.grad.data)

    nccl_allreduce_by_buckets(nc, kn, all_grads)
