import os
import sys
import time

# TODO: the following environments should be set out of here
#os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
#os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
#os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
from numba import cuda
from numba.cuda import driver as cuda_driver

if not os.environ.get('NCCL_DEBUG'):
    os.environ['NCCL_DEBUG'] = 'WARN'

if not os.environ.get('NUMBA_NCCLLIB'):
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

# NOTE: for IB, different with nodes
#if not os.environ.get('NCCL_SOCKET_IFNAME'):
#    os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'
# TODO: or
#export NCCL_IB_DISABLE=1

# TODO: delete when pynccl install
#PYNCCL_DIR = os.environ.get('PYNCCL_PATH')
#if not PYNCCL_DIR:
#    PYNCCL_DIR = '/data/likun/proj/py/pygu/cuda/pynccl'
#sys.path.insert(0, PYNCCL_DIR)

import pynccl
from pynccl import Nccl, NcclWrp
from pynccl import binding as pynccl_binding


'''
# Data types
ncclInt8       = 0; ncclChar       = 0
ncclUint8      = 1
ncclInt32      = 2; ncclInt        = 2
ncclUint32     = 3
ncclInt64      = 4
ncclUint64     = 5
ncclFloat16    = 6; ncclHalf       = 6
ncclFloat32    = 7; ncclFloat      = 7
ncclFloat64    = 8; ncclDouble     = 8
ncclNumTypes   = 9
'''
PYNCCL_DTYPE_MAP = {
    'char': pynccl_binding.ncclChar,
    'int8': pynccl_binding.ncclInt8,
    'uint8': pynccl_binding.ncclUint8,
    'int': pynccl_binding.ncclInt,
    'int32': pynccl_binding.ncclInt32,
    'uint32': pynccl_binding.ncclUint32,
    'long': pynccl_binding.ncclInt64,
    'int64': pynccl_binding.ncclInt64,
    'uint64': pynccl_binding.ncclUint64,
    'float16': pynccl_binding.ncclFloat16,
    'half': pynccl_binding.ncclHalf,
    'float': pynccl_binding.ncclFloat,
    'float32': pynccl_binding.ncclFloat32,
    'float64': pynccl_binding.ncclFloat64,
    'double': pynccl_binding.ncclDouble,
}


'''
# Reduction operation selector
ncclSum        = 0
ncclProd       = 1
ncclMax        = 2
ncclMin        = 3
ncclNumOps     = 4
'''
PYNCCL_REDUCE_OP_MAP = {
    'sum': pynccl_binding.ncclSum,
    'prod': pynccl_binding.ncclProd,
    'max': pynccl_binding.ncclMax,
    'min': pynccl_binding.ncclMin,
    'numops': pynccl_binding.ncclNumOps,
}
