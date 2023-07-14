"""Pytorch Tensor Utils"""
from __future__ import print_function

import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ----------------------------------------------------------------------------
# tensor stack / split

def pt_ts_vstack(seq):
    '''[3,4] -> [2,3,4] -> [6,4]'''
    shp = seq[0].shape
    shp = list(shp)
    v = torch.stack(seq, dim=0)
    shp[0] = shp[0] * len(seq)
    r = v.reshape(shp)
    return r


def pt_ts_hstack(seq):
    '''[3,4] -> [3,2,4] -> [3,8]'''
    shp = seq[0].shape
    shp = list(shp)
    v = torch.stack(seq, dim=-2)
    nshp = shp[:-1] + [-1]  # [shp[-1] * shp[-2]]
    r = v.reshape(nshp)
    return r


def pt_ts_vsplit(a, n):
    '''[4,6] -> [2,6] x2'''
    z = a.shape[0] // n
    r = torch.split(a, z, dim=0)
    return r


def pt_ts_hsplit(a, n):
    '''[4,6] -> [4,3] x2'''
    z = a.shape[-1] // n
    r = torch.split(a, z, dim=-1)
    return r


def pt_ts_copy(x, requires_grad=None):
    is_cuda = x.is_cuda
    device = x.device
    if requires_grad is None:
        requires_grad = x.requires_grad

    if is_cuda:
        a = x.cpu().detach().numpy()
    else:
        a = x.detach().numpy()
    y = torch.tensor(a, requires_grad=requires_grad).to(device)
    return y


def pt_is_tensor(x):
    return isinstance(x, torch.Tensor)
