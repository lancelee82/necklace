"""Shared Memory Wrapper using Pytroch Tensor"""
import os
import sys

import numpy as np

import torch

import weakref


# <1> cuda tensor
# TODO: error when call torch.cuda.FloatStorage._new_shared_cuda(*kb)

'''
(vpt3) likun@gpu44:/data/likun/proj/py/pydl/projs/pt/tts/sbspt_tacotron$ python
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> a = torch.randn((3,4))
>>> a
tensor([[-0.2036,  0.6665,  1.5075, -1.1996],
        [-0.1722, -0.3881, -0.5578,  1.0250],
        [ 1.0348, -0.3037, -2.3502,  0.6826]])
>>> b = a.cuda(0)
sb = b.storage()


>>> sb = b.storage()
>>>
>>>
>>> kb = sb._share_cuda_()
>>> kb
(0, b'p\xb5\x8c\x03......\x01\xd0\xc1\x9e\x03\x00\\', 262144, 865280, 12)
>>> sbs = torch.cuda.FloatStorage._new_shared_cuda(*kb)
THCudaCheck FAIL file=/pytorch/torch/csrc/generic/StorageSharing.cpp line=304 error=10 : invalid device ordinal
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: cuda runtime error (10) : invalid device ordinal at /pytorch/torch/csrc/generic/StorageSharing.cpp:304
>>> sb.is_shared()
True

>>> c = torch.cuda.FloatTensor(sb)
>>> c
tensor([-0.2036,  0.6665,  1.5075, -1.1996, -0.1722, -0.3881, -0.5578,
         1.0250,  1.0348, -0.3037, -2.3502,  0.6826], device='cuda:0')
>>> c.reshape(3,4)
tensor([[-0.2036,  0.6665,  1.5075, -1.1996],
        [-0.1722, -0.3881, -0.5578,  1.0250],
        [ 1.0348, -0.3037, -2.3502,  0.6826]], device='cuda:0')
>>> c
tensor([-0.2036,  0.6665,  1.5075, -1.1996, -0.1722, -0.3881, -0.5578,
         1.0250,  1.0348, -0.3037, -2.3502,  0.6826], device='cuda:0')
>>>
'''

# ==========================================================================
# <2.1> [create] cpu tensor
'''
>>> import torch
>>> a = torch.randn((3,4))
>>> a
tensor([[ 1.8132, -0.4170,  0.5468, -1.2568],
        [-0.1466, -1.2931, -1.8427, -1.5484],
        [-0.3124, -0.8293,  0.7023,  2.7585]])
>>> sa = a.storage()
>>> sa
 1.8131818771362305
 -0.416991651058197
 0.5468318462371826
 -1.2568247318267822
 -0.14656424522399902
 -1.2930991649627686
 -1.8427164554595947
 -1.548437237739563
 -0.3123556077480316
 -0.8293309807777405
 0.7022885084152222
 2.758500576019287
[torch.FloatStorage of size 12]
>>> fd, sz = sa._share_fd_()
>>> fd
3
>>> sz
12
>>> type(fd)
<class 'int'>
>>>


>>> b = torch.randn((3,4))
>>> b
tensor([[ 2.0740,  0.2943, -0.3822, -0.2354],
        [ 0.5352, -0.1917,  0.2583, -0.8923],
        [-0.1868,  0.1278, -0.8274,  0.0092]])
>>> sb = b.storage()
>>>
>>> mb = sb._share_filename_()
>>> mb
(b'/tmp/filejgS3Md', b'/torch_5819_770656832', 12)
>>> sb._shared_incref()
>>>
>>>
'''

# <2.2> [get] cpu tensor
'''
>>> import torch
>>> sa = torch.FloatStorage._new_shared_fd(3, 12)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: could not duplicate a shared memory file descriptor
>>>


>>> sb = torch.FloatStorage._new_shared_filename(b'/tmp/filejgS3Md', b'/torch_5819_770656832', 12)
>>> sb
 2.073974370956421
 0.29428884387016296
 -0.3821811378002167
 -0.23536504805088043
 0.5352218747138977
 -0.1916685849428177
 0.2583489418029785
 -0.8923362493515015
 -0.186796635389328
 0.12776611745357513
 -0.8273906707763672
 0.009174988605082035
[torch.FloatStorage of size 12]
>>> sbb = sb._shared_decref()
>>> sbb
 2.073974370956421
 0.29428884387016296
 -0.3821811378002167
 -0.23536504805088043
 0.5352218747138977
 -0.1916685849428177
 0.2583489418029785
 -0.8923362493515015
 -0.186796635389328
 0.12776611745357513
 -0.8273906707763672
 0.009174988605082035
[torch.FloatStorage of size 12]
>>> b = torch.FloatTensor(sbb)
>>> b
tensor([ 2.0740,  0.2943, -0.3822, -0.2354,  0.5352, -0.1917,  0.2583,
        -0.8923, -0.1868,  0.1278, -0.8274,  0.0092])
>>> b.shape
torch.Size([12])
>>> b = b.reshape((3,4))
>>> b
tensor([[ 2.0740,  0.2943, -0.3822, -0.2354],
        [ 0.5352, -0.1917,  0.2583, -0.8923],
        [-0.1868,  0.1278, -0.8274,  0.0092]])
>>> b.shape
torch.Size([3, 4])
>>>
'''


# NOTE: now only support cpu pytorch tensor

# pytorch dtype and tensor/storage dict
TORCH_TENSOR_TYPE_D = {
    'torch.uint8': [torch.ByteTensor, torch.ByteStorage,
                  torch.cuda.ByteTensor, torch.cuda.ByteStorage],
    'torch.int8': [torch.CharTensor, torch.CharStorage,
                 torch.cuda.CharTensor, torch.cuda.CharStorage],
    'torch.int16': [torch.ShortTensor, torch.ShortStorage,
                  torch.cuda.ShortTensor, torch.cuda.ShortStorage],
    'torch.int32': [torch.IntTensor, torch.IntStorage,
                  torch.cuda.IntTensor, torch.cuda.IntStorage],
    'torch.int': [torch.IntTensor, torch.IntStorage,
                torch.cuda.IntTensor, torch.cuda.IntStorage],
    'torch.int64': [torch.LongTensor, torch.LongStorage,
                  torch.cuda.LongTensor, torch.cuda.LongStorage],
    'torch.long': [torch.LongTensor, torch.LongStorage,
                 torch.cuda.LongTensor, torch.cuda.LongStorage],
    'torch.float32': [torch.FloatTensor, torch.FloatStorage,
                    torch.cuda.FloatTensor, torch.cuda.FloatStorage],
    'torch.float': [torch.FloatTensor, torch.FloatStorage,
                  torch.cuda.FloatTensor, torch.cuda.FloatStorage],
    'torch.float64': [torch.DoubleTensor, torch.DoubleStorage,
                    torch.cuda.DoubleTensor, torch.cuda.DoubleStorage],
    'torch.long': [torch.DoubleTensor, torch.DoubleStorage,
                 torch.cuda.DoubleTensor, torch.cuda.DoubleStorage],
}


def dtype_to_tensor(dtype, use_cuda=False):
    if use_cuda:
        return TORCH_TENSOR_TYPE_D[dtype][2]
    else:
        return TORCH_TENSOR_TYPE_D[dtype][0]


def dtype_to_storage(dtype, use_cuda=False):
    if use_cuda:
        return TORCH_TENSOR_TYPE_D[dtype][3]
    else:
        return TORCH_TENSOR_TYPE_D[dtype][1]


# from /pytorch/torch/multiprocessing/reductions.py

class SharedMemArray(object):
    def __init__(self, name=None, obj_shape=(1,), obj_cnt=1,
                 dtype='torch.float', gpu_i=0, sk=None, use_cuda=False):
        self.name = name
        self.obj_shape = obj_shape
        self.obj_cnt = obj_cnt
        self.dtype = dtype
        self.gpu_i = gpu_i
        self.sk = sk
        self.use_cuda = use_cuda

        if self.use_cuda:
            torch.cuda._lazy_init()

        self.init_shms()
        self.init_bufs(self.sk)

    def init_shms(self):
        pass

    def init_bufs(self, sk=None):
        if sk is None:
            sk, b = self.create_mx_ndarray_shmem(self.obj_shape, dtype=self.dtype)
            self.sk = sk
            self.buf = b
        else:
            b = self.ndarray_from_shmem_key(sk, shape=self.obj_shape, dtype=self.dtype)
            self.buf = b

    # TODO: if self.use_cuda:

    # for cuda tensor
    @classmethod
    def ndarray_from_shmem_key_cuda_(cls, sk, shape, dtype='torch.float'):
        storage = torch.cuda.FloatStorage._new_shared_cuda(*sk)  # TODO:
        b = torch.cuda.FloatTensor(storage)  # TODO: by dtype
        b = b.reshape(shape)
        return b

    @classmethod
    def create_mx_ndarray_shmem_cuda_(cls, shape, dtype='torch.float'):
        a = torch.zeros(shape, dtype=dtype)
        storage = a.storage()
        sk = storage._share_cuda_()
        b = a
        return sk, b

    # for cpu tensor
    @classmethod
    def ndarray_from_shmem_key(cls, sk, shape, dtype='torch.float'):
        stg_cls = dtype_to_storage(dtype)
        storage = stg_cls._new_shared_filename(*sk)
        storage = storage._shared_decref()
        tsr_cls = dtype_to_tensor(dtype)
        b = tsr_cls(storage)
        b = b.reshape(shape)
        return b

    @classmethod
    def create_mx_ndarray_shmem(cls, shape, dtype='torch.float'):
        a = torch.zeros(shape)#, dtype=dtype)  # TODO:
        storage = a.storage()
        sk = storage._share_filename_()
        sa = storage._shared_incref()
        b = a
        return sk, b

    @classmethod
    def create_from_shmem_key(cls, sk=None, shape=(1,), dtype='torch.float'):
        if not sk:
            raise Exception('no shmem key given')
        a = cls(sk=sk, obj_shape=shape, dtype=dtype)
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
