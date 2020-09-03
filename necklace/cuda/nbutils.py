"""Numba Utils"""
from __future__ import print_function

import ctypes as ctp

import numpy as np

from numba import cuda


def cuda_select_device(dev_i):
    try:
        cuda.close()
    except Exception as e:
        print(e)
        #pass
    cuda.select_device(dev_i)


def cuda_current_context():
    ctx = cuda.current_context()
    return ctx


def cuda_stream():
    stream = cuda.stream()
    return stream


def cuda_to_device(A, stream=None):
    if not stream:
        stream = cuda_stream()
    dA = cuda.to_device(A, stream)
    return dA


def cuda_devarr_c_void_p(d_arr):  # TODO: d_arr.as_cuda_arg(...)
    p_arr = ctp.c_void_p(d_arr.device_ctypes_pointer.value)
    return p_arr

