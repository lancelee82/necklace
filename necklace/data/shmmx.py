"""Shared Memory Wrapper using Mxnet NDArray"""
import os

import numpy as np

import mxnet as mx


class SharedMemArray(object):
    def __init__(self, name=None, obj_shape=(1,), obj_cnt=1,
                 dtype=np.float32, sk=None):
        self.name = name
        self.obj_shape = obj_shape
        self.obj_cnt = obj_cnt
        self.dtype = dtype
        self.sk = sk

        self.init_shms()
        self.init_bufs(self.sk)

    def init_shms(self):
        pass

    def init_bufs(self, sk=None):
        if sk is None:
            shape = self.obj_shape
            dtype = self.dtype
            sk, b = self.create_mx_ndarray_shmem(shape, dtype=dtype)
            self.sk = sk
            self.buf = b
        else:
            b = self.ndarray_from_shmem_key(sk)
            self.buf = b

    @classmethod
    def ndarray_from_shmem_key(cls, sk):
        b = mx.nd.NDArray(mx.nd.ndarray._new_from_shared_mem(*sk))
        return b

    @classmethod
    def create_mx_ndarray_shmem(cls, shape, dtype=np.float32):
        # NOTE: this is not the right way to create a ndarray shared memory
        # that will cause the inferrence count error when process ends
        #a = mx.nd.zeros(shape, dtype=dtype)
        #sk = a._to_shared_mem()
        #print(sk)
        #b = cls.ndarray_from_shmem_key(sk)

        skk = (-1, -1, shape, dtype)
        a = mx.nd.NDArray(mx.nd.ndarray._new_from_shared_mem(*skk))
        sk = a._to_shared_mem()
        b = a

        return sk, b

    @classmethod
    def create_from_shmem_key(cls, sk=None):
        if not sk:
            raise Exception('no shmem key given')
        a = cls(sk=sk)
        return a

    # should be called by creater when to release shared memory
    def unlink(self):
        del self.buf

    def __del__(self):
        pass

    def __getitem__(self, idx):
        o = self.buf[idx]
        return o

    def __setitem__(self, idx, item):
        self.buf[idx] = item
