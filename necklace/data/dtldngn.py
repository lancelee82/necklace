"""Dataloader Wrapper"""
from __future__ import absolute_import
import six

from . import sampler as _sampler


class DataLoaderNgnBase(object):

    # for distinguishing from other dataloader
    _attr_ngn_dataloader = True

    def __init__(self, dataset,
                 batch_size=None,
                 shuffle=False,
                 sampler=None,
                 last_batch='discard',
                 batch_sampler=None,
                 batchify_fn=None,
                 *args, **kwargs):
        super(DataLoaderNgnBase, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.last_batch = last_batch
        self.batch_sampler = batch_sampler
        self.batchify_fn = batchify_fn

        if not self.sampler:
            self.init_slice_sampler()

        if not self.batch_sampler:
            self.batch_sampler = self._make_batch_sampler(self.sampler)

    def init_slice_sampler(self):
        ds_len = len(self.dataset)
        self.sampler = _sampler.SliceSampler(int(ds_len), shuffle=self.shuffle)

    def _make_batch_sampler(self, smplr):
        batch_sampler = _sampler.BatchSampler(
            smplr, self.batch_size,
            self.last_batch if self.last_batch else 'keep')
        return batch_sampler

    # for distributed dataloader
    def reset_batch_sampler(self, new_sampler, *args, **kwargs):
        self.sampler = new_sampler
        self.batch_sampler = self._make_batch_sampler(self.sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for batch in self.batch_sampler:
            b = self._get_a_batch(batch)
            if self.batchify_fn:
                dt = self.batchify_fn(b)
            else:
                dt = b
            yield dt

    def _get_a_batch(self, batch):
        b = []
        for t in batch:
            a = self.dataset[t]
            b.append(a)
        return b
