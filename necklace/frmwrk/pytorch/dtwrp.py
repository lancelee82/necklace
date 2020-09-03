"""Dataloader Wrapper"""
import torch
import torch.utils.data
import torch.utils.data.distributed


class DataLoaderIterWrp(torch.utils.data._DataLoaderIter):
    def __init__(self, *args, **kwargs):
        super(DataLoaderIterWrp, self).__init__(*args, **kwargs)

    def _put_indices(self):
        pass

    def _put_indices_lazy(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1


class DataLoaderWrp(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderWrp, self).__init__(*args, **kwargs)

    def __iter__(self):
        return DataLoaderIterWrp(self)
