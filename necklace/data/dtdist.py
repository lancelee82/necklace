""""""
import copy
from collections import OrderedDict

import numpy as np

from necklace.cuda.ncclwrp import PYNCCL_DTYPE_MAP
from necklace.cuda import ncclutils
from necklace.cuda import ncclgrp
from necklace.cuda import ncclfns


def dt_sampler_do_nccl_bcast(nc, kn, root, buf):
    buf_p = ncclutils.tensor_data_ptr(buf)
    sz = ncclutils.tensor_size(buf)

    nc.do_bcast(buf_p, sz, root, datatype=PYNCCL_DTYPE_MAP['uint32'])

    nc.stream_sync()

    return buf


def dt_sampler_do_nccl_allreduce(nc, kn, root, buf):
    buf_p = ncclutils.tensor_data_ptr(buf)
    sz = ncclutils.tensor_size(buf)

    nc.do_all_reduce(buf_p, buf_p, sz, datatype=PYNCCL_DTYPE_MAP['uint32'])

    nc.stream_sync()

    return buf


def dt_shape_do_nccl_broadcast(pg, shp):
    if pg is None:
        pg = ncclgrp.get_nccl_group_mp_or_main()

    pg.stream_sync()
    ncclfns.broadcast(shp, root=0, group=pg)
    pg.stream_sync()


def dt_input_do_nccl_broadcast(pg, dt, dtype='float32'):
    if pg is None:
        pg = ncclgrp.get_nccl_group_mp_or_main()

    pg.stream_sync()
    ncclfns.broadcast(dt, root=0, datatype=PYNCCL_DTYPE_MAP[dtype], group=pg)
    pg.stream_sync()

