"""DataLoader using Multi Processes and Multi Data Queue"""
from __future__ import absolute_import
import six

import copy
import random
import time
from random import randint

import multiprocessing as mp
import multiprocessing.queues

import numpy as np
import numpy.random as npr

from . import sampler as _sampler


# NOTE: if use this dataloader, you should do multiprcessing reduction firstly,
# such as case of pytorch, you should do following before creating dataloader:
#from torch import multiprocessing as torch_mp
# in which the function of init_reductions() is called.


class DataLoaderSharedMemory(object):

    # for distinguishing from other dataloader
    _attr_ngn_dataloader = True

    """Dataloader Control Entry"""
    def __init__(self, dataset, batch_size=None,
                 data_shape=None,
                 shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None,
                 batchify_fn=None,  # for self batch
                 inter_batchify_fn=None,  # for internal dataloader
                 part_num=20,  # for part loader
                 num_workers=0):
        self.dataset = dataset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.last_batch = last_batch
        self.batch_sampler = batch_sampler

        self.num_workers = num_workers
        if self.num_workers < 1:
            raise Exception('num_workers should > 0')

        if batchify_fn is None:
            raise Exception('no batchify_fn is specified')
        else:
            self.batchify_fn = batchify_fn

        self.inter_batchify_fn = inter_batchify_fn

        self.num = None  # num
        if self.num is None:
            self.num = len(self.dataset) // self.batch_size

        self.cache_n = self.num_workers  # cache_n
        self.cache_i = 0
        self.cache_num = None  # cache_num
        if self.cache_num is None:
            self.init_cache_num()

        self.part_num = part_num  # num of batches in shared memory once time

        self.init_qs()

        self.init_sampler()
        self.init_loaders()

    def init_cache_num(self):
        # NOTE: self.cache_n > 0
        n = self.num // self.cache_n
        m = self.num % self.cache_n
        if m == 0:
            self.cache_num = n
        else:
            self.cache_num = n + 1

    def init_qs(self):
        # TODO: set queue size
        self.qs_msg = [mp.Queue() for i in range(self.cache_n)]
        self.qs_data = [mp.Queue() for i in range(self.cache_n)]
        self.qs_ctl = [mp.Queue() for i in range(self.cache_n)]

    def init_sampler(self):
        if self.batch_sampler is None:
            if self.batch_size is None:
                raise ValueError("batch_size must be specified unless " \
                                 "batch_sampler is specified")
            if self.sampler is None:
                self.init_slice_sampler()
            elif self.shuffle:
                raise ValueError("shuffle must not be specified if sampler is specified")

            self.batch_sampler = _sampler.BatchSampler(
                self.sampler, self.batch_size,
                self.last_batch if self.last_batch else 'keep')

        elif self.batch_size is not None or self.shuffle or self.sampler is not None or \
                self.last_batch is not None:
            raise ValueError("batch_size, shuffle, sampler and last_batch must " \
                             "not be specified if batch_sampler is specified.")

    def init_slice_sampler(self):
        ds_len = self.num * self.batch_size  #len(self.dataset)
        self.sampler = _sampler.SliceSampler(int(ds_len), shuffle=self.shuffle)

    def get_one_batch_sampler(self, i):
        a = self.cache_num * self.batch_size * i
        b = self.cache_num * self.batch_size * (i + 1)
        s = self.sampler[int(a):int(b)]
        bs = _sampler.SliceBatchSampler(s, self.batch_size)
        return bs

    def reset_batch_sampler(self, new_sampler):
        self.sampler = new_sampler
        self.batch_sampler = _sampler.BatchSampler(
            self.sampler, self.batch_size,
            self.last_batch if self.last_batch else 'keep')

        self.num = len(new_sampler) // self.batch_size
        self.init_cache_num()

        for i in range(self.cache_n):
            bs = self.get_one_batch_sampler(i)
            self.qs_ctl[i].put(('reset_batch_sampler', bs))

    def get_one_loader(self, i):
        bs = self.get_one_batch_sampler(i)
        ldr = DataloaderShmem(self.dataset, bs, self.batch_size,
                              batchify_fn=self.inter_batchify_fn)
        return ldr

    def start_one_loader_cache(self, i, q_msg, q_data, q_ctl,
                               ldr, num, part_num, shuffle):
        cchp = DataloaderShmemPart(
            i, q_msg, q_data, q_ctl, ldr,
            num=num, part_num=part_num, shuffle=shuffle)
        return cchp

    def init_loaders(self):
        self.op_loaders = []
        self.workers = []

        for i in range(self.cache_n):
            ldr = self.get_one_loader(i)
            self.op_loaders.append(ldr)

            worker = mp.Process(
                target=self.start_one_loader_cache,
                args=(i, self.qs_msg[i], self.qs_data[i], self.qs_ctl[i],
                      ldr, len(ldr), self.part_num, self.shuffle))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def next_cache(self):
        self.cache_i += 1
        self.cache_i %= self.cache_n

    def __len__(self):
        return self.num

    def _get_a_batch(self, dt):
        d = self.batchify_fn(dt)
        return d

    def __iter__(self):
        # clear qs
        self.clear_qs()

        # NOTE: ask loader to start load data
        self.qs_ctl_put(('start_epoch', 1))

        # begin from the first cache
        self.cache_i = 0

        q_get_timeout_n = 0

        num_batches = self.num

        curr_idx = 0
        while curr_idx < num_batches:
            #print('>>>> ', self.cache_i, curr_idx, num_batches)

            #n = self.qs_msg[self.cache_i].get()

            # NOTE: use timeout to avoid blocking forever
            '''
            try:
                if curr_idx == 0:  # wait with no timeout at the first time
                    tmout = None
                else:
                    # NOTE: dynamically set the time
                    tmout = self.part_num * (3 + q_get_timeout_n)
                n = self.qs_msg[self.cache_i].get(timeout=tmout)
            except Exception as e:
                if q_get_timeout_n <= self.cache_n:
                    n = 0
                    q_get_timeout_n += 1
                else:
                    break  # no data
            '''

            try:
                if curr_idx == 0:  # wait with no timeout at the first time
                    tmout = None
                else:
                    # NOTE: dynamically set the time
                    tmout = 0.01
                n = self.qs_msg[self.cache_i].get(timeout=tmout)
            except Exception as e:
                n = 0  # goto next cache

            for j in range(n):
                dt = self.qs_data[self.cache_i].get()  # (data, label)
                data_batch = self._get_a_batch(dt)
                curr_idx += 1
                yield data_batch  # >>> iter entry

            if n > 0:  # continue load data
                self.qs_ctl[self.cache_i].put(('get_items', self.part_num))  # continue load data

            self.next_cache()

        for k in range(self.cache_n):
            self.qs_ctl[k].put(('stop_epoch', 0))

    def clear(self):  # NOTE: should be called outside when main process ends

        self.qs_ctl_put(('clear', 0))

        for worker in self.workers:
            worker.join()

    def q_clear(self, q):
        while not q.empty():
            try:
                a = q.get(timeout=1)
            except Exception as e:
                pass

    def qs_clear(self, qs):
        for q in qs:
            self.q_clear(q)

    def clear_qs(self):
        self.qs_clear(self.qs_msg)
        self.qs_clear(self.qs_data)
        self.qs_clear(self.qs_ctl)

    def qs_ctl_put(self, msg):
        for q in self.qs_ctl:
            q.put(msg)


class DataloaderShmem(object):
    """Simple dataloader for one process"""
    def __init__(self, dataset, batch_sampler, batch_size, batchify_fn=None,
                 last_batch=None, *args, **kwargs):
        self._dataset = dataset
        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self._batchify_fn = batchify_fn
        if self._batchify_fn is None:
            self._batchify_fn = self._default_batchify_fn

    def _default_batchify_fn(self, dt):
        return dt

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for batch in self.batch_sampler:
            yield self._batchify_fn([self._dataset[idx] for idx in batch])
        return

    def reset(self):
        pass

    def shuffle(self):
        self.batch_sampler.shuffle()

    def reset_batch_sampler(self, new_batch_sampler):
        self.batch_sampler = new_batch_sampler


class DataloaderShmemPart(object):
    """Dataloader process for part"""
    def __init__(self, cache_i, q_msg, q_data, q_ctl,
                 dataloader, num=None, part_num=20, shuffle=True,
                 *args, **kwargs):
        self.cache_i = cache_i
        self.q_msg = q_msg
        self.q_data = q_data
        self.q_ctl = q_ctl

        self.dataloader = dataloader

        self.batch_size = self.dataloader.batch_size

        self.num = num
        if self.num is None:
            self.num = len(self.dataloader)

        self.part_num = part_num
        self.init_part_n()

        self.shuffle = shuffle

        self.cache = {}

        self.loop()

    def init_part_n(self):
        n = self.num / self.part_num
        m = self.num % self.part_num
        if m == 0:
            self.part_n = n
        else:
            self.part_n = n + 1

    def loop(self):
        while 1:
            msg = self.q_ctl.get()
            cmd = msg[0]
            if cmd == 'start_epoch':
                ret = self.load()
                cmd = ret[0]
                if cmd == 'clear':
                    break
            elif cmd == 'get_items':
                pass
            elif cmd == 'stop_epoch':
                pass
            elif cmd == 'clear':
                break
            elif cmd == 'reset_batch_sampler':
                bs = msg[1]
                self.reset_batch_sampler(bs)
            else:
                raise Exception('unknown cmd %s' % str(msg))

    def load(self):
        if self.shuffle:
            self.dataloader.shuffle()

        nbatch = 0

        part_num = self.part_num

        data_iter = iter(self.dataloader)
        end_of_batch = False

        try:
            # pre fetch next batch
            next_data_batch = next(data_iter)
        except StopIteration:
            end_of_batch = True
        except Exception as e:
            print('nx'*100, e)
            end_of_batch = True

        i = 0
        while not end_of_batch:
            #print('>', self.cache_i, nbatch)

            #data_batch = next_data_batch
            #data, label = data_batch

            try:
                #self.q_data.put((data, label))
                self.q_data.put(next_data_batch)

                i += 1
            except Exception as e:
                print('x' * 40, e)
                pass  # next data

            if i % part_num == 0:
                self.q_msg.put(i)
                i = 0

                msg = self.q_ctl.get()
                cmd = msg[0]
                if cmd == 'stop_epoch':
                    end_of_batch = True
                elif cmd == 'get_items':
                    part_num = msg[1]
                else:  # clear
                    return msg

            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
            except StopIteration:
                end_of_batch = True

            nbatch += 1

            if not end_of_batch:
                if nbatch >= self.num:
                    end_of_batch = True

        # the last data
        if i > 0:
            self.q_msg.put(i)
            i = 0

        # wait control msg from q
        while 1:
            try:
                msg = self.q_ctl.get(timeout=100)
            except Exception as e:
                msg = ('stop_epoch', 0)

            return msg

    def __len__(self):
        return self.num

    def reset(self):
        self.dataloader.reset()

    def reset_batch_sampler(self, new_batch_sampler):
        self.dataloader.reset_batch_sampler(new_batch_sampler)
