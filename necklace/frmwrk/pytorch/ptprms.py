""""""
import copy
from collections import OrderedDict

import numpy as np

import torch

#from necklace.cuda import nbutils
from necklace.cuda import ncclgrp
from necklace.cuda import ncclfns
from necklace.cuda.ncclwrp import PYNCCL_DTYPE_MAP


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
# nccl utils

def nccl_nc_allreduce(nc, p_arr_send, p_arr_recv, sz):
    nc.stream_sync()
    nc.do_all_reduce(p_arr_send, p_arr_recv, sz)
    nc.stream_sync()


def nccl_pg_allreduce(pg, p_arr_send, p_arr_recv, sz):
    if pg is None:
        pg = ncclgrp.get_nccl_group_dp()  # NOTE: default is dp_pg
    if pg is None:
        pg = ncclgrp.get_nccl_group_main()  # NOTE: or the main_pg
    #print('nccl_pg_allreduce:', pg.to_cfg_dict())
    pg.stream_sync()
    ncclfns.pg_all_reduce(p_arr_send, p_arr_recv, sz, group=pg)
    pg.stream_sync()


def nccl_pg_broadcast(pg, p_arr_send, p_arr_recv, sz, root):
    if pg is None:
        pg = ncclgrp.get_nccl_group_dp()  # NOTE: default is dp_pg
    if pg is None:
        pg = ncclgrp.get_nccl_group_main()  # NOTE: or the main_pg
    #print('nccl_pg_broadcast:', pg.to_cfg_dict())
    pg.stream_sync()
    ncclfns.pg_broadcast(p_arr_send, p_arr_recv, sz, root, group=pg)
    pg.stream_sync()


def nccl_pg_reduce(pg, p_arr_send, p_arr_recv, sz, root):
    if pg is None:
        pg = ncclgrp.get_nccl_group_dp()  # NOTE: default is dp_pg
    if pg is None:
        pg = ncclgrp.get_nccl_group_main()  # NOTE: or the main_pg
    #print('nccl_pg_reduce:', pg.to_cfg_dict())
    pg.stream_sync()
    ncclfns.pg_reduce(p_arr_send, p_arr_recv, sz, root, group=pg)
    pg.stream_sync()


# -------------------------------------------------------------------------
# for params do nccl

# ===================================================
# TODO: use flatten / buckets like nccl grads
# ===================================================


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

        if not p.requires_grad:
            continue

        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = p.data
            #d_arr_recv = prm_revs[name]

            sz = np.prod(d_arr_send.size()) #* d_arr_send.element_size()
            # <1.1>
            '''
            nc.do_all_reduce(d_arr_send.data_ptr(),
                             d_arr_send.data_ptr(),
                             sz)
            nc.stream_sync()
            '''
            # <1.2>
            '''
            nccl_nc_allreduce(nc,
                         d_arr_send.data_ptr(),
                         d_arr_send.data_ptr(),
                         sz)
            # <2>
            '''
            nccl_pg_allreduce(nc, #None,  # dp_pg
                              d_arr_send.data_ptr(),
                              d_arr_send.data_ptr(),
                              sz)

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


def params_do_nccl_broadcast(nc, kn, root, model, prm_revs=None):

    for name, p in model.named_parameters():

        if not p.requires_grad:
            continue

        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = p.data
            #d_arr_recv = prm_revs[name]

            sz = np.prod(d_arr_send.size()) #* d_arr_send.element_size()

            nccl_pg_broadcast(nc, #None,  # dp_pg
                              d_arr_send.data_ptr(),
                              d_arr_send.data_ptr(),
                              sz, root)

            #d_arr_send[:] = d_arr_send / float(kn)

    # for debug
    #print('param:', p)


def params_fltn_do_nccl_broadcast(nc, kn, root, d_arr_send, sz):

    nccl_pg_broadcast(nc, #None,  # dp_pg
                      d_arr_send.data_ptr(),
                      d_arr_send.data_ptr(),
                      sz, root)

    # for debug
    #print('param:', p)


# -------------------------------------------------------------------------
# flatten tensors for do nccl with good performance

from torch._utils import (_flatten_dense_tensors,
                          _unflatten_dense_tensors, 
                          _take_tensors)


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

        sz = np.prod(grads_batch_coalesced.size())

        # NOTE:
        #nbutils.cuda_current_context().synchronize()
        # or,
        # <1.1>
        '''
        nc.stream_sync()
        nc.do_all_reduce(grads_batch_coalesced.data_ptr(),
                         grads_batch_coalesced.data_ptr(),
                         sz)
        nc.stream_sync()
        '''
        # <1.2>
        '''
        nccl_nc_allreduce(nc,
                     grads_batch_coalesced.data_ptr(),
                     grads_batch_coalesced.data_ptr(),
                     sz)
        # <2>
        '''
        nccl_pg_allreduce(nc, #None,  # dp_pg
                     grads_batch_coalesced.data_ptr(),
                     grads_batch_coalesced.data_ptr(),
                     sz)

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


# -------------------------------------------------------------------------
# for grads reduce to me (for ZeRO opt)

def opt_grads_do_nccl_reduce(nc, kn, root, opt, prm_revs=None, grad_comp_cfg=None):
    if isinstance(opt, (list, tuple)):
        for o in opt:
            opt_one_grads_do_nccl_reduce(
                nc, kn, root, o, prm_revs=prm_revs, grad_comp_cfg=grad_comp_cfg)
    else:
        opt_one_grads_do_nccl_reduce(
            nc, kn, root, opt, prm_revs=prm_revs, grad_comp_cfg=grad_comp_cfg)


def opt_one_grads_do_nccl_reduce(nc, kn, root, opt, prm_revs=None, grad_comp_cfg=None):

    all_grads = []

    for group in opt.param_groups:
        for p in group['params']:
            if p.grad is not None:
                all_grads.append(p.grad.data)

    nccl_reduce_by_buckets(nc, kn, root, all_grads)

    # for debug
    #print('grads:', p.grad)


def nccl_reduce_by_buckets(nc, kn, root, all_grads):
    # Now bucketing the parameters
    dev_grads_buckets = _take_tensors(all_grads, nccl_reduce_bucket_size)

    for grads_batch in dev_grads_buckets:
        grads_batch_coalesced = _flatten_dense_tensors(grads_batch)
        #print('grads_batch_coalesced', grads_batch_coalesced.device)

        # NOTE:
        torch.cuda.synchronize()

        sz = np.prod(grads_batch_coalesced.size())

        nccl_pg_reduce(nc, #None,  # dp_pg
                       grads_batch_coalesced.data_ptr(),
                       grads_batch_coalesced.data_ptr(),
                       sz, root)

        #grads_batch_coalesced[:] = grads_batch_coalesced / float(kn)

        grads_batch_reduced = _unflatten_dense_tensors(
            grads_batch_coalesced, grads_batch)

        for grad, reduced in zip(grads_batch, grads_batch_reduced):
            grad.copy_(reduced)


def opt_grads_fltn_do_nccl_reduce(nc, kn, root, d_arr_send, sz, grad_comp_cfg=None):

    nccl_pg_reduce(nc, #None,  # dp_pg
                   d_arr_send.data_ptr(),
                   d_arr_send.data_ptr(),
                   sz, root)


# -------------------------------------------------------------------------
# for trans outputs-inputs and grads

def init_tensors_recv(shps, ctx=None, dtype=torch.float32):
    ts = []
    for shp in shps:
        a = torch.zeros(shp, dtype=dtype)
        if ctx is not None:
            a = a.to(ctx)
        ts.append(a)
    return ts


def tensors_do_nccl_p2p_send(nc, rank_send, rank_recv, send_tensors, dtype='float'):

    for p in send_tensors:

        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = p.data.data_ptr()
            d_arr_recv = None

            sz = np.prod(p.data.size())
            nc.do_pp_send_recv(d_arr_send, d_arr_recv, sz,
                               rank_send, rank_recv,
                               datatype=PYNCCL_DTYPE_MAP[dtype])

            nc.stream_sync()


def tensors_do_nccl_p2p_recv(nc, rank_send, rank_recv, recv_tensors, dtype='float'):

    for p in recv_tensors:

        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = None
            d_arr_recv = p.data.data_ptr()

            sz = np.prod(p.data.size())
            nc.do_pp_send_recv(d_arr_send, d_arr_recv, sz,
                               rank_send, rank_recv,
                               datatype=PYNCCL_DTYPE_MAP[dtype])

            nc.stream_sync()

