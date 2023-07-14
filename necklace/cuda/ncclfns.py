"""NCCL Functions"""
#
# NOTE: 1. a pynccl wrapper for ease of use
#       2. use with ncclgrp
#
# TODO: 1. other nccl functions (like in pynccl.api)
#       2. check ncclutils.py
#       3. datatype is None ==> get it by torch dtype (nccluitls.py: tensor_dtype())
#
# ----------------------------------------------------------------------------

from ctypes import POINTER, byref

import numpy as np

import torch

from .ncclwrp import pynccl
from .ncclwrp import Nccl
from .ncclwrp import NcclWrp
from .ncclutils import tensor_data_ptr, tensor_dtype


def nk_comm_init_rank(nk, comm_i, world_size, nuid, rank):
    r = nk.comm_init_rank(byref(comm_i), world_size, nuid, rank)
    return comm_i


# ----------------------------------------------------------------------------
# all_gather

def nccl_fn_on_comm_all_gather(nk, comm_i, pg_rank, pg_ranks,
                               p_arr_send, p_arr_recv, sz,
                               datatype=None):
    if pg_rank not in pg_ranks:
        return  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ranks = sorted(pg_ranks)
    kn = len(ranks)
    rank = ranks.index(pg_rank)

    stream_i = nk.get_stream()

    r = nk.group_start()
    #print(rank, '>>> ncclGroupStart ', r)

    if datatype is None:
        datatype = pynccl.binding.ncclFloat

    r = nk.all_gather(p_arr_send, p_arr_recv,
                      sz, datatype,
                      comm_i, stream_i.handle)
    #print(rank, '>>> ncclAllGather ', r)

    r = nk.group_end()
    #print(rank, '>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    # not here
    #r = nk.comm_destroy(comm_i)
    #print(rank, '>>> ncclCommDestroy ', r)


def all_gather(output, input_, datatype=None, group=None):
    sz = np.prod(input_.size())

    p_arr_send = tensor_data_ptr(input_)
    p_arr_recv = tensor_data_ptr(output)

    nccl_fn_on_comm_all_gather(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, datatype=datatype)


def pg_all_gather(p_arr_send, p_arr_recv, sz, datatype=None, group=None):
    nccl_fn_on_comm_all_gather(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, datatype=datatype)


# ----------------------------------------------------------------------------
# all_reduce

def nccl_fn_on_comm_all_reduce(nk, comm_i, pg_rank, pg_ranks,
                               p_arr_send, p_arr_recv, sz,
                               datatype=None,
                               op=None):
    if pg_rank not in pg_ranks:
        return  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ranks = sorted(pg_ranks)
    kn = len(ranks)
    rank = ranks.index(pg_rank)

    stream_i = nk.get_stream()

    r = nk.group_start()
    #print(rank, '>>> ncclGroupStart ', r)

    if datatype is None:
        datatype = pynccl.binding.ncclFloat
    if op is None:
        op = pynccl.binding.ncclSum

    r = nk.all_reduce(p_arr_send, p_arr_recv,
                      sz, datatype, op,
                      comm_i, stream_i.handle)  # NOTE:
                      #comm_i, c_void_p(0))  # NOTE:
    #print(rank, '>>> ncclAllReduce ', r)

    r = nk.group_end()
    #print(rank, '>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    # not here
    #r = nk.comm_destroy(comm_i)
    #print(rank, '>>> ncclCommDestroy ', r)


def all_reduce(input_, datatype=None, op=None, group=None):
    sz = np.prod(input_.size())

    p_arr_send = tensor_data_ptr(input_)
    p_arr_recv = tensor_data_ptr(input_)

    nccl_fn_on_comm_all_reduce(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, datatype=datatype, op=op)


def pg_all_reduce(p_arr_send, p_arr_recv, sz, datatype=None, op=None, group=None):
    nccl_fn_on_comm_all_reduce(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, datatype=datatype, op=op)


# ----------------------------------------------------------------------------
# broadcast

def nccl_fn_on_comm_broadcast(nk, comm_i, pg_rank, pg_ranks,
                               p_arr_send, p_arr_recv, sz, root,
                               datatype=None):
    if pg_rank not in pg_ranks:
        print('necklace: ncclfns: nccl_fn_on_comm_broadcast() do nothing !', pg_rank, pg_ranks)
        return  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ranks = sorted(pg_ranks)
    kn = len(ranks)
    rank = ranks.index(pg_rank)

    stream_i = nk.get_stream()

    r = nk.group_start()
    #print(rank, '>>> ncclGroupStart ', r)

    if datatype is None:
        datatype = pynccl.binding.ncclFloat

    r = nk.broadcast(p_arr_send, p_arr_recv,
                     sz, datatype, root,
                     comm_i,
                     stream_i.handle)
    #print(rank, '>>> ncclBroadcast ', r)

    r = nk.group_end()
    #print(rank, '>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    # not here
    #r = nk.comm_destroy(comm_i)
    #print(rank, '>>> ncclCommDestroy ', r)


def broadcast(input_, root=None, datatype=None, group=None):
    if root is None:
        root = group.rank

    sz = np.prod(input_.size())
    p_arr_send = tensor_data_ptr(input_)
    p_arr_recv = tensor_data_ptr(input_)

    nccl_fn_on_comm_broadcast(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, root, datatype=datatype)

    #print('broadcast:', group.to_cfg_dict(), sz)


def pg_broadcast(p_arr_send, p_arr_recv, sz, root=None, datatype=None, group=None):
    if root is None:
        root = group.rank
    nccl_fn_on_comm_broadcast(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, root, datatype=datatype)


# ----------------------------------------------------------------------------
# reduce

def nccl_fn_on_comm_reduce(nk, comm_i, pg_rank, pg_ranks,
                           p_arr_send, p_arr_recv, sz, root,
                           datatype=None,
                           op=None):
    if pg_rank not in pg_ranks:
        return  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ranks = sorted(pg_ranks)
    kn = len(ranks)
    rank = ranks.index(pg_rank)

    stream_i = nk.get_stream()

    r = nk.group_start()
    #print(rank, '>>> ncclGroupStart ', r)

    if datatype is None:
        datatype = pynccl.binding.ncclFloat
    if op is None:
        op = pynccl.binding.ncclSum

    r = nk.reduce(p_arr_send, p_arr_recv,
                  sz, datatype, op, root,
                  comm_i,
                  stream_i.handle)
    #print(rank, '>>> reduce ', r)

    r = nk.group_end()
    #print(rank, '>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    # not here
    #r = nk.comm_destroy(comm_i)
    #print(rank, '>>> ncclCommDestroy ', r)


def reduce(input_, root=None, datatype=None, op=None, group=None):
    if root is None:
        root = group.rank

    sz = np.prod(input_.size())
    p_arr_send = tensor_data_ptr(input_)
    p_arr_recv = tensor_data_ptr(input_)

    nccl_fn_on_comm_reduce(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, root, datatype=datatype, op=op)

    #print('reduce:', group.to_cfg_dict(), sz)


def pg_reduce(p_arr_send, p_arr_recv, sz, root=None, datatype=None, op=None, group=None):
    if root is None:
        root = group.rank
    nccl_fn_on_comm_reduce(
        group.nk, group.comm_i, group.pg_rank, group.pg_ranks,
        p_arr_send, p_arr_recv, sz, root, datatype=datatype, op=op)


