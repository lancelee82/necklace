"""Dataload Dummy for MP"""


class DtstDummy(object):
    def __init__(self, ds_len):
        self.ds_len = ds_len

    def __getitem__(self):
        return None

    def __len__(self):
        return self.ds_len


class DtldDummy(object):
    def __init__(self, ds, batch_size, drop_last=True):
        self.dataset = ds
        self.ds_len = len(ds)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.iter_len = self.ds_len // batch_size
        if not drop_last:
            if self.ds_len % batch_size != 0:
                self.iter_len += 1

    def __iter__(self):
        for i in range(self.iter_len):
            yield (None, None)  # (input, target)

    def __len__(self):
        return self.iter_len
