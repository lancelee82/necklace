import numpy as np

try:
    import mxnet
    from mxnet.gluon.data import dataloader
except:
    mxnet = None

try:
    import torch
    from torch.utils.data import dataloader
except:
    torch = None


def nothing_batchify_fn(dt):
    return dt


def label_reduce_batchify_fn(dt, lbl_n=1000):
    for i in range(len(dt[1])):
        dt[1][i][0] %= lbl_n
    return dt


if torch:
    def np2pt_batchify_fn(dt):
        data = torch.from_numpy(dt[0])
        label = torch.from_numpy(dt[1].astype(np.long))
        dt = (data, label)
        return dt


    def pt_batchify_fn(dt):
        d = dataloader.default_collate(dt)
        return d


    def flt2int_bchfn(dt):
        data, label = dt[0], dt[1]
        #lbl = label.type(torch.int32)
        lbl = label.type(torch.long)  # int64
        dt = (data, lbl)
        return dt


if mxnet:
    def mx_batchify_fn(dt):
        d = dataloader.default_mp_batchify_fn(dt)
        return d
