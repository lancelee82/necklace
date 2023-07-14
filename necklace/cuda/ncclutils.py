"""NCCL utils"""
# NOTE: some functions are not used and to be cleaned
# TODO: the functions whose framework-dependent should be moved to frmwrk dir

import six

import numpy as np


# -----------------------------------------------------------------
# TODO: del and use common nccl func

def seg_cb_nccl_ar(og, nc, kn, ndarray):
    """Seg Output Grad All-Reduce by NCCL"""
    if og is None:
        return og

    ndarray.waitall()  # NOTE: this wait is important for sync

    sz = og.size
    nc.do_all_reduce(og.get_data_p(),
                     og.get_data_p(),
                     sz)

    #og[:] = og / float(kn)

    nc.stream_sync()

    ndarray.waitall()

    return og


def nccl_out_all_gather(out, outs, nc, kn, ndarray):
    """Output All-Gather by NCCL"""
    ndarray.waitall()

    sz = out.size
    nc.do_all_gather(out.get_data_p(),
                     outs.get_data_p(),
                     sz)

    nc.stream_sync()

    ndarray.waitall()

    return outs


# -----------------------------------------------------------------
# TODO: 
# 1. a.get_data_p() ==> a.data_ptr()  : mxnet same with pytorch
# 2. def tensor_size():

def tensor_data_ptr(a):

    # for mxnet.ndarray
    #p_arr_send = d_arr_send.get_data_p()
    #p_arr_recv = d_arr_recv.get_data_p()
    # for numba.cuda.devicearray
    #p_arr_send = d_arr_send.device_ctypes_pointer
    #p_arr_recv = d_arr_recv.device_ctypes_pointer
    # do getting ctypes_pointer outside

    if hasattr(a, 'get_data_p'):
        # mxnet
        return a.get_data_p()
    elif hasattr(a, 'data_ptr'):
        # pytorch
        return a.data_ptr()
    elif hasattr(a, 'device_ctypes_pointer'):
        # numba.cuda.devicearray
        return a.device_ctypes_pointer
    else:
        return None


def tensor_size(a):
    if not hasattr(a, 'size'):
        return None

    sz = getattr(a, 'size')

    if isinstance(sz, six.integer_types + (float,)):
        # mxnet
        return sz
    elif callable(sz):
        s = sz()
        if isinstance(s, six.integer_types + (float,)):
            return s
        elif isinstance(s, (tuple, list)):
            # pytorch
            return np.prod(s)  # or, a.numel()
        else:
            return None
    else:
        return None


"""
>>> for v in dir(torch):
...     if 'Tensor' in v:
...         print(v)
... 
BFloat16Tensor
BoolTensor
ByteTensor
CharTensor
DoubleTensor
FloatTensor
HalfTensor
IntTensor
LongTensor
ShortTensor
Tensor
TensorType
>>> 
"""
def tensor_dtype(a):
    pass  # TODO:


# -----------------------------------------------------------------
# TODO: mxnet [o]  pytorch [t]

def nccl_all_reduce(og, nc, kn, ndarray=None):
    """Seg Output Grad All-Reduce by NCCL"""
    if og is None:
        return og

    if ndarray:
        ndarray.waitall()  # NOTE: this wait is important for sync
        #t#og.wait_to_read()

    sz = og.size
    nc.do_all_reduce(og.get_data_p(),
                     og.get_data_p(),
                     sz)

    #og[:] = og / float(kn)

    nc.stream_sync()

    if ndarray:
        ndarray.waitall()
        #t#og.wait_to_read()

    return og


def nccl_all_gather(out, outs, nc, kn, ndarray=None):
    """Output All-Gather by NCCL"""

    if ndarray:
        ndarray.waitall()
        #t#out.wait_to_read()

    sz = out.size
    nc.do_all_gather(out.get_data_p(),
                     outs.get_data_p(),
                     sz)

    nc.stream_sync()

    if ndarray:
        ndarray.waitall()
        #t#out.wait_to_read()

    return outs


def nccl_reduce(buf, root, nc, kn, ndarray=None):
    if ndarray:
        ndarray.waitall()
        #t#buf.wait_to_read()

    sz = buf.size
    nc.do_reduce(buf.get_data_p(),
                 buf.get_data_p(),
                 sz, root)

    #buf[:] = buf / float(kn)

    nc.stream_sync()

    if ndarray:
        ndarray.waitall()
        #t#buf.wait_to_read()

    return buf


def nccl_bcast(buf, root, nc, kn, ndarray=None):
    if ndarray:
        ndarray.waitall()
        #t#buf.wait_to_read()

    sz = buf.size
    nc.do_bcast(buf.get_data_p(),
                sz, root)

    nc.stream_sync()

    if ndarray:
        ndarray.waitall()
        #t#buf.wait_to_read()

    return buf


def nccl_reduce_scatter(sendbuf, recvbuf, nc, kn, ndarray=None):
    if ndarray:
        ndarray.waitall()
        #t#sendbuf.wait_to_read()

    recvcount = recvbuf.size
    nc.do_reduce_scatter(sendbuf.get_data_p(),
                         recvbuf.get_data_p(),
                         recvcount, root)

    #recvbuf[:] = recvbuf / float(kn)

    nc.stream_sync()

    if ndarray:
        ndarray.waitall()
        #t#recvbuf.wait_to_read()

    return recvbuf
