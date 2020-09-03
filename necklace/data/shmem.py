import os
import time

try:
    import cPickle as pickle
except:
    import pickle

import mmap

import numpy as np
import posix_ipc


class SharedMemList(object):
    def __init__(self, name=None, obj_size=1, obj_cnt=1):
        self.name = name
        self.obj_size = obj_size
        self.obj_cnt = obj_cnt

        size = obj_size * obj_cnt
        if name:
            self._shm = posix_ipc.SharedMemory(name)
        else:
            self._shm = posix_ipc.SharedMemory(None, posix_ipc.O_CREX, size=size)

        self._buf = mmap.mmap(self._shm.fd, size)

    @classmethod
    def copy(cls, arr):
        return cls(obj_size=arr.obj_size, obj_cnt=arr.obj_cnt)

    @classmethod
    def zeros_like(cls, arr):
        return cls(obj_size=arr.obj_size, obj_cnt=arr.obj_cnt)

    @classmethod
    def bin_size(cls, o):
        s = cls.pck_dumps(o)
        siz = len(s)
        return siz

    @classmethod
    def pck_dumps(self, o):
        s = pickle.dumps(o)
        return s

    @classmethod
    def pck_loads(self, s):
        o = pickle.loads(s)
        return o

    def unlink(self):
        self._shm.unlink()

    def __del__(self):
        self._buf.close()
        self._shm.close_fd()


    def write_to_memory(self, o, sk=0):
        self._buf.seek(sk)
        self._buf.write(o)

    def read_from_memory(self, z=0, sk=0):
        self._buf.seek(sk)
        if z:
            o = self._buf.read(z)
        else:
            o = self._buf.read()

        return o

    def __getitem__(self, idx):
        sk = self.obj_size * idx
        s = self.read_from_memory(self.obj_size, sk=sk)
        o = self.pck_loads(s)
        return o

    def __setitem__(self, idx, item):
        s = self.pck_dumps(item)
        sk = self.obj_size * idx
        self.write_to_memory(s, sk=sk)


class SharedMemArray(object):
    def __init__(self, name=None, obj_size=1, obj_cnt=1):
        self.name = name
        self.obj_size = obj_size
        self.obj_cnt = obj_cnt

        self.init_shms()
        self.init_bufs()
        self.init_sizes()

    def init_shms(self):
        self.shms = []
        size = self.obj_size
        for i in range(self.obj_cnt):
            if self.name:
                _shm = posix_ipc.SharedMemory(self.name)
            else:
                _shm = posix_ipc.SharedMemory(None, posix_ipc.O_CREX, size=size)
            self.shms.append(_shm)

    def init_bufs(self):
        self.bufs = []
        size = self.obj_size
        for i in range(self.obj_cnt):
            _shm = self.shms[i]
            _buf = mmap.mmap(_shm.fd, size)
            self.bufs.append(_buf)

    def init_sizes(self):
        #self.sizes = [0] * self.obj_cnt
        shape = (self.obj_cnt,)
        dtype = np.int32#np.float32
        name = self.name

        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if name:
            self._shm_size = posix_ipc.SharedMemory(name)
        else:
            self._shm_size = posix_ipc.SharedMemory(None, posix_ipc.O_CREX, size=size)
        self._buf_size = mmap.mmap(self._shm_size.fd, size)
        self.sizes = np.ndarray(shape, dtype, self._buf_size, order='C')

    @classmethod
    def copy(cls, arr):
        return cls(obj_size=arr.obj_size, obj_cnt=arr.obj_cnt)

    @classmethod
    def zeros_like(cls, arr):
        return cls(obj_size=arr.obj_size, obj_cnt=arr.obj_cnt)

    @classmethod
    def bin_size(cls, o):
        s = cls.pck_dumps(o)
        siz = len(s)
        return siz

    @classmethod
    def pck_dumps(self, o):
        s = pickle.dumps(o)
        return s

    @classmethod
    def pck_loads(self, s):
        o = pickle.loads(s)
        return o

    def unlink(self):
        for i in range(self.obj_cnt):
            self.shms[i].unlink()

        self._shm_size.unlink()

    def __del__(self):
        for i in range(self.obj_cnt):
            self.bufs[i].close()
            self.shms[i].close_fd()

        self._buf_size.close()
        self._shm_size.close_fd()

    def write_to_memory(self, _buf, o, sk=0):
        _buf.seek(sk)
        _buf.write(o)
        _buf.flush()

    def read_from_memory(self, _buf, z=0, sk=0):
        _buf.seek(sk)
        o = _buf.read(z)
        return o

    def __getitem__(self, idx):
        _buf = self.bufs[idx]
        sz = self.sizes[idx]
        s = self.read_from_memory(_buf, sz, sk=0)
        o = self.pck_loads(s)
        return o

    def __setitem__(self, idx, item):
        _buf = self.bufs[idx]
        s = self.pck_dumps(item)
        self.sizes[idx] = len(s)
        self.write_to_memory(_buf, s, sk=0)
