"""MPI funcs those are compatible with NCCL in ptprms.py"""
import copy
from collections import OrderedDict

import numpy as np

import torch

'''
#from necklace.cuda import nbutils
from necklace.cuda import ncclgrp
from necklace.cuda import ncclfns
from necklace.cuda.ncclwrp import PYNCCL_DTYPE_MAP
'''


# -------------------------------------------------------------------------
# nccl utils

def nccl_nc_allreduce(nc, p_arr_send, p_arr_recv, sz):
    nc.stream_sync()
    nc.do_all_reduce(p_arr_send, p_arr_recv, sz)
    nc.stream_sync()


'''
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
'''





def init_params_recvs(model):
    prm_revs = OrderedDict()

    for name, p in model.named_parameters():
        print(name)

        if not p.requires_grad:
            continue

        #if p.grad is not None:  # or p.requires_grad
        shp = p.shape
        if len(shp) >= 1 and shp[0] > 0:
            a = torch.zeros(shp)
            if p.is_cuda:
                a = a.to(p.device)
            prm_revs[name] = a

    return prm_revs


def init_opt_grads_recvs(opt):
    prm_revs = OrderedDict()

    for j, group in enumerate(opt.param_groups):
        for i, p in enumerate(group['params']):
            name = str(j) + str(i)

            if not p.requires_grad:
                continue

            #if p.grad is not None:  # or p.requires_grad
            shp = p.shape
            #print(name, shp)
            if len(shp) >= 1 and shp[0] > 0:
                a = torch.zeros(shp)
                #if p.is_cuda:
                #    a = a.to(p.device)
                prm_revs[name] = a

    return prm_revs



def params_do_mpi_allreduce(nc, kn, model, prm_revs=None):

    for name, p in model.named_parameters():
        #print(name)

        if not p.requires_grad:
            continue

        shp = p.shape
        #print(name, shp)
        if len(shp) >= 1 and shp[0] > 0:

            d_arr_send = p.detach().cpu().numpy()
            d_arr_recv = prm_revs[name].detach().cpu().numpy()

            sz = np.prod(d_arr_send.size) #* d_arr_send.element_size()
            # <1.1>
            nc.do_all_reduce(d_arr_send,
                             d_arr_recv,
                             sz)
            nc.stream_sync()
            '''
            # <1.2>
            nccl_nc_allreduce(nc,
                              d_arr_send.data_ptr(),
                              d_arr_send.data_ptr(),
                              sz)
            # <2>
            '''
            '''
            nccl_pg_allreduce(nc, #None,  # dp_pg
                              d_arr_send.data_ptr(),
                              d_arr_send.data_ptr(),
                              sz)
            '''

            #d_arr_send[:] = d_arr_send / float(kn)
            p.data[:] = torch.from_numpy(d_arr_recv) / float(kn)





def opt_grads_do_mpi_allreduce(nc, kn, opt, prm_revs=None, grad_comp_cfg=None):

    for j, group in enumerate(opt.param_groups):
        for i, p in enumerate(group['params']):
            name = str(j) + str(i)

            if p.grad is None:  # or not p.requires_grad
                continue

            #if p.grad is not None:  # or p.requires_grad
            shp = p.grad.shape
            #print(name, shp)
            if len(shp) >= 1 and shp[0] > 0:

                d_arr_send = p.grad.detach().cpu().numpy()
                d_arr_recv = prm_revs[name].detach().cpu().numpy()

                sz = np.prod(d_arr_send.size) #* d_arr_send.element_size()
                # <1.1>
                nc.do_all_reduce(d_arr_send,
                                 d_arr_recv,
                                 sz)
                nc.stream_sync()

                p.grad.data[:] = torch.from_numpy(d_arr_recv) / float(kn)





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
