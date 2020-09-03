"""DataLoader using Multi Processes and Shared Memory (mxnet.ndarray)"""
import six

import copy
import random
import time
from random import randint

import multiprocessing as mp
import multiprocessing.queues

import numpy as np
import numpy.random as npr

import sampler as _sampler
# shared memory pool using mxnet.ndarray
import shmmx as shdnda


class DataLoaderSharedMemory(object):
    """Dataloader Control Entry"""
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 inter_batchify_fn=None,  # for internal dataloader
                 part_num=20,  # for part loader
                 num_workers=0):
        self._dataset = dataset

        # NOTE: the samplers created here are not used now
        if batch_sampler is None:
            if batch_size is None:
                raise ValueError("batch_size must be specified unless " \
                                 "batch_sampler is specified")
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError("shuffle must not be specified if sampler is specified")

            batch_sampler = _sampler.BatchSampler(
                sampler, batch_size, last_batch if last_batch else 'keep')
        elif batch_size is not None or shuffle or sampler is not None or \
                last_batch is not None:
            raise ValueError("batch_size, shuffle, sampler and last_batch must " \
                             "not be specified if batch_sampler is specified.")

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        if batchify_fn is None:
            raise Exception('no batchify_fn is specified')
        else:
            self.batchify_fn = batchify_fn

        self.inter_batchify_fn = inter_batchify_fn

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num = None  # num
        if self.num is None:
            self.num = len(self._dataset) / self.batch_size

        self.cache_n = self._num_workers  # cache_n
        self.cache_i = 0
        self.cache_num = None  # cache_num
        if self.cache_num is None:
            self.init_cache_num()

        self.data, self.label = self._dataset[0]  # for data info

        self.batch_data_shape = (self.batch_size,) + self.data.shape
        self.batch_label_shape = (self.batch_size,)

        self.part_num = part_num  # num of batches in shared memory once time

        self.init_qs()

        # NOTE: the shared memory and sub-processes will alive through all epochs
        self.init_data_shm((self.part_num,) + self.batch_data_shape, np.float32, None)
        self.init_label_shm((self.part_num,) + self.batch_label_shape, np.float32, None)

        self.init_sampler()
        self.init_loaders()

    def init_cache_num(self):
        if self.cache_n <= 0:
            self.cache_num = self.num
        else:
            n = self.num / self.cache_n
            m = self.num % self.cache_n
            if m == 0:
                self.cache_num = n
            else:
                self.cache_num = n + 1

    def init_data_shm(self, shape, typ, name):
        # <3> mxnet ndarray shared
        self.data_shm = [shdnda.SharedMemArray(name, obj_shape=shape, dtype=typ) for i in range(self._num_workers)]

    def init_label_shm(self, shape, typ, name):
        self.label_shm = [shdnda.SharedMemArray(name, obj_shape=shape, dtype=typ) for i in range(self._num_workers)]

    def init_qs(self):
        # TODO: set queue size
        self.qs_msg = [mp.queues.Queue() for i in range(self.cache_n)]
        self.qs_data = [mp.queues.Queue() for i in range(self.cache_n)]
        self.qs_ctl = [mp.queues.Queue() for i in range(self.cache_n)]

    def init_sampler(self):
        self._sampler = _sampler.SliceSampler(len(self._dataset), self.shuffle)

    def get_one_batch_sampler(self, i):
        a = self.cache_num * self.batch_size * i
        b = self.cache_num * self.batch_size * (i + 1)
        s = self._sampler[a:b]
        bs = _sampler.SliceBatchSampler(s, self.batch_size)
        return bs

    def get_one_loader(self, i):
        bs = self.get_one_batch_sampler(i)
        ldr = DataloaderShmem(self._dataset, bs, self.batch_size,
                              batchify_fn=self.inter_batchify_fn)
        return ldr

    def start_one_loader_cache(self, i, q_msg, q_data, q_ctl,
                               data_shm, label_shm, ldr,
                               num, part_num, shuffle):
        cchp = DataloaderShmemPart(
            i, q_msg, q_data, q_ctl, data_shm, label_shm, ldr,
            num=num, part_num=part_num, shuffle=shuffle)
        return cchp

    def init_loaders(self):
        self.caches = []

        for i in range(self.cache_n):
            ldr = self.get_one_loader(i)

            worker = mp.Process(
                target=self.start_one_loader_cache,
                args=(i, self.qs_msg[i], self.qs_data[i], self.qs_ctl[i],
                      self.data_shm[i].sk, self.label_shm[i].sk,  # NOTE: only pass the shmem key
                      ldr, len(ldr), self.part_num, self.shuffle))
            worker.daemon = True
            worker.start()
            self.caches.append(worker)

    def next_cache(self):
        self.cache_i += 1
        self.cache_i %= self.cache_n

    def _get_a_batch(self, dt):
        d = self.batchify_fn(dt)
        return d

    def __iter__(self):
        # clear qs
        self.clear_qs()

        # NOTE: ask loader to start load data
        self.qs_ctl_put('start')

        # begin from the first cache
        self.cache_i = 0

        q_get_timeout_n = 0

        num_batches = self.num

        curr_idx = 0
        while curr_idx < num_batches:
            #n = self.qs_msg[self.cache_i].get()
            # NOTE: use timeout to avoid blocking forever
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

            for j in range(n):
                data = self.data_shm[self.cache_i][j]
                label = self.label_shm[self.cache_i][j]
                data_batch = self._get_a_batch((data, label))
                curr_idx += 1
                yield data_batch  # >>> iter entry

            self.qs_ctl[self.cache_i].put(self.part_num)  # continue load data
            self.next_cache()

        for k in range(self.cache_n):
            self.qs_ctl[k].put(0)

    def clear(self):  # NOTE: should be called outside when main process ends

        self.qs_ctl_put('stop')

        for worker in self.caches:
            worker.join()

        # NOTE: delete ndarray shared memory manually
        self.shm_unlink()

    def shm_unlink(self):
        for shm in self.data_shm:
            shm.unlink()

        for shm in self.label_shm:
            shm.unlink()

    def __len__(self):
        return len(self._batch_sampler)

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

    def qs_ctl_put(self, msg='start'):
        for q in self.qs_ctl:
            q.put(msg)


class DataloaderShmem(object):
    """Simple dataloader for one process"""
    def __init__(self, dataset, batch_sampler, batch_size, batchify_fn=None,
                 last_batch=None, *args, **kwargs):
        self._dataset = dataset
        self._batch_sampler = batch_sampler
        self.batch_size = batch_size
        self._batchify_fn = batchify_fn
        if self._batchify_fn is None:
            self._batchify_fn = self._default_batchify_fn

    def _default_batchify_fn(self, dt):
        return dt

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        for batch in self._batch_sampler:
            yield self._batchify_fn([self._dataset[idx] for idx in batch])
        return

    def reset(self):
        pass

    def shuffle(self):
        self._batch_sampler.shuffle()


class DataloaderShmemPart(object):
    """Dataloader process for part"""
    def __init__(self, cache_i, q_msg, q_data, q_ctl, data_shm, label_shm,
                 dataloader, num=None, part_num=20, shuffle=True,
                 *args, **kwargs):
        self.cache_i = cache_i
        self.q_msg = q_msg
        self.q_data = q_data
        self.q_ctl = q_ctl
        self.data_shm_sk = data_shm
        self.label_shm_sk = label_shm

        self.dataloader = dataloader

        self.batch_size = self.dataloader.batch_size

        self.num = num
        if self.num is None:
            self.num = len(self.dataloader)

        self.part_num = part_num
        self.init_part_n()

        self.shuffle = shuffle

        self.init_shm()

        self.cache = {}

        self.loop()

    def init_part_n(self):
        n = self.num / self.part_num
        m = self.num % self.part_num
        if m == 0:
            self.part_n = n
        else:
            self.part_n = n + 1

    def init_shm(self):
        self.data_shm = shdnda.SharedMemArray.create_from_shmem_key(self.data_shm_sk)
        self.label_shm = shdnda.SharedMemArray.create_from_shmem_key(self.label_shm_sk)

    def loop(self):
        while 1:
            msg = self.q_ctl.get()
            if msg == 'start':
                ret = self.load()
                if ret == 'stop':
                    break
            elif msg == 'stop':
                break

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

        i = 0
        while not end_of_batch:
            #print('>', self.cache_i, nbatch)

            data_batch = next_data_batch
            # TODO: multi dims data
            data, label = data_batch
            try:
                # NOTE: maybe the shapes of data and buf are not the same
                self.data_shm[i][:] = data
                self.label_shm[i][:] = label
                i += 1
            except Exception as e:
                pass  # next data

            if i % part_num == 0:
                self.q_msg.put(i)
                i = 0

                msg = self.q_ctl.get()
                if isinstance(msg, six.string_types):
                    #if msg == 'stop':
                    #    return msg
                    #else:
                    #    pass  # TODO: other cmds
                    return msg
                elif isinstance(msg, six.integer_types):
                    if msg == 0:
                        #return
                        end_of_batch = True
                    else:
                        part_num = msg

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
                msg = self.q_ctl.get(timeout=10)
            except Exception as e:
                msg = 0

            return msg

    def __len__(self):
        return self.num

    def reset(self):
        self.dataloader.reset()


class DataLoaderShmForModule(DataLoaderSharedMemory):
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 inter_batchify_fn=None,  # for internal dataloader
                 part_num=20,  # for part loader
                 num_workers=0):
        super(DataLoaderShmForModule, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            last_batch=last_batch, batch_sampler=batch_sampler,
            batchify_fn=batchify_fn,
            inter_batchify_fn=inter_batchify_fn,
            part_num=part_num,
            num_workers=num_workers)

        # TODO: for provide_data
        self.mode = 'train'
        #self.data, self.label = self._dataset[0]

    @property
    def provide_data(self):
        if self.mode == 'train':
            return [('data', (self.batch_size,) + self.data.shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        if self.mode == 'train':
            return [('label', (self.batch_size, ))]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    def reset(self):
        pass
