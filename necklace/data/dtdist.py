""""""
import copy
from collections import OrderedDict

import numpy as np

from necklace.cuda import ncclutils
from pynccl import binding as pynccl_binding


def dt_sampler_do_nccl_bcast(nc, kn, root, buf):
    buf_p = ncclutils.tensor_data_ptr(buf)
    sz = ncclutils.tensor_size(buf)

    nc.do_bcast(buf_p, sz, root, datatype=pynccl_binding.ncclUint32)

    nc.stream_sync()

    return buf


def dt_sampler_do_nccl_allreduce(nc, kn, root, buf):
    buf_p = ncclutils.tensor_data_ptr(buf)
    sz = ncclutils.tensor_size(buf)

    nc.do_all_reduce(buf_p, buf_p, sz, datatype=pynccl_binding.ncclUint32)

    nc.stream_sync()

    return buf

