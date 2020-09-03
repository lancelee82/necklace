""""""
import random


class Dataset(object):
    """Abstract dataset class. All datasets should have this interface.

    Subclasses need to override `__getitem__`, which returns the i-th
    element, and `__len__`, which returns the total number elements.

    .. note:: An mxnet or numpy array can be directly used as a dataset.
    """
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CacheDataset(Dataset):
    def __init__(self, dt, num=None, cache_num=None, transform=None):
        self.dt = dt
        self.num = num
        if self.num is None:
            self.num = len(self.dt)
        self.cache_num = cache_num
        self.transform = transform

        self.cache = {}
        self.cache_i = 0

    def __getitem__(self, idx):
        d = self.cache_get(idx)

        if d is None:
            d = self.one_item(idx)
            if self.transform:
                d = self.transform(d)
            self.cache_put(idx, d)

        return d

    def __len__(self):
        return self.num

    def cache_get(self, idx):
        if not self.cache_num:
            return None

        d = self.cache.get(idx)
        return d

    def cache_put(self, idx, d):
        if not self.cache_num:
            return None

        self.cache[idx] = d
        self.cache_i += 1

        if self.cache_i > self.cache_num:
            n = self.cache_i - self.cache_num
            self.cache_del_rand(self.cache_i - self.cache_num)
            self.cache_i -= n

    def cache_del_rand(self, n):
        for i in range(n):
            k = random.choice(self.cache.keys())
            del self.cache[k]

    #overwrite
    def one_item(self, idx, *args, **kwargs):
        d = self.dt[idx]
        return d
