"""Data Utils"""

import itertools
import types
import random

import numpy as np



# TODO: from snipe, to merge to necklace



# #########################################################################
# data utils
# #########################################################################

def is_generator(a):
    return isinstance(a, types.GeneratorType)


def is_iterator(a):
    #x#if hasattr(a, '__iter__') and hasattr(a, 'next'):  ==> generator
    if hasattr(a, '__iter__'):
        return True
    return False


def get_len(a):
    if hasattr(a, '__len__'):
        return len(a)
    else:
        raise Exception('No __len__ defined')


def np_is_ndarray(d):
    if isinstance(d, np.ndarray):
        return True
    else:
        return False


def np_array(d):
    if np_is_ndarray(d):
        return d

    a = np.array(d)
    return a


def label_to_onehot(y, shp):
    o = np.zeros(shp)
    r = shp[0]
    yi = np.array(y, dtype='int')
    o[np.arange(r), yi] = 1.0
    return o


# #########################################################################
# data loader for distributed training
# #########################################################################

# when distributed training, the data should be split to parts for workers,
# and the shuffle should be done firstly at a centric place, then send the
# shuffle result to the workers at train init or send each part at each batch.

# data index sampler
# data index batch slicer


class SamplerWrp(object):
    def __init__(self, length, shuffle=False):
        self._length = length
        self.indices = list(range(self._length))

        # NOTE: this will take a long time when length is big
        # TODO: use numpy will fast [?]
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self._length

    def __getslice__(self, i, j):
        return self.indices[i:j]


# from mxnet.gluon.data.dataset.py
class DatasetBase(object):
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DatasetWrp(DatasetBase):
    def __init__(self, ds, *args, **kwargs):
        super(DatasetWrp, self).__init__()
        self.ds = ds

    def __getitem__(self, idx):
        r = [d[idx] for d in self.ds]
        return r

    def __len__(self):
        return len(self.ds[0])


class DataLoaderBase(object):
    def __init__(self):
        pass

    # NOTE: if this method is implemented, the dataloader wrapper
    # will call it directly and return without any other process
    #def get_batch_by_dbi(self, dbi, dbf=None):
    #    pass


class DataLoaderWrp(object):
    def __init__(self, loader, ds, rank, mdtype, role,
                 batch_size=None, workers=None, shuffle=False,
                 *args, **kwargs):
        self.loader = loader
        self.ds = ds
        self.rank = rank
        self.mdtype = mdtype
        self.role = role
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle

        # TODO: last_batch='discard'

        self.init_sampler()


    def init_sampler(self):
        ln = len(self.ds)
        self.sampler = SamplerWrp(ln, self.shuffle)


    def get_data_batch_index(self, rank, i, b=None, k=None):
        if b is None:
            b = self.batch_size
        if k is None:
            k = self.workers

        i1 = i * (b * k) + rank * b
        i2 = i * (b * k) + (rank + 1) * b

        dbi = self.sampler[i1:i2]
        return dbi


    def get_data_batch(self, dbi, dbf=None):
        if hasattr(self.loader, 'get_batch_by_dbi'):
            batch = self.loader.get_batch_by_dbi(dbi, dbf)
        else:
            batch = self._get_data_batch_from_ds(dbi, dbf)
        return batch


    def _get_data_batch_from_ds(self, dbi, dbf=None):
        batch = [self.ds[idx] for idx in dbi]
        # NOTE: then should call the mdwrp.data_batchify(batch)
        if dbf is not None:
            batch = dbf(batch)
        return batch
