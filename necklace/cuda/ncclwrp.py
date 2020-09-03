import os
import sys
import time

import numpy as np


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
