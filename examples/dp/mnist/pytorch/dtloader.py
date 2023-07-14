

# TODO: 

class MnistDataLoader(gluon.data.DataLoader):
    def __init__(self, dataset, ti, tn, batch_size=64, shuffle=True,
                 batch_sampler=None, batchify_fn=None,
                 num_workers=0):
        self._dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.ti = ti
        self.tn = tn

        self.init_sampler()
        batch_sampler = self.get_one_batch_sampler(self.ti)

        super(MnistDataLoader, self).__init__(
            dataset, #batch_size=batch_size, shuffle=shuffle,
            batch_sampler=batch_sampler, batchify_fn=batchify_fn,
            num_workers=num_workers)

    def reset(self):
        pass

    def clear(self):
        pass

    def init_sampler(self):
        self._sampler = _sampler.SliceSampler(len(self._dataset), self.shuffle)

    def get_one_batch_sampler(self, i):
        part_n = len(self._dataset) / self.tn
        a = int(part_n * i)
        b = int(part_n * (i + 1))
        s = self._sampler[a:b]
        bs = _sampler.SliceBatchSampler(s, self.batch_size)
        return bs
