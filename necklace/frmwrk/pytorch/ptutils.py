"""Pytorch Utils"""

import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_grad_norm(model, clip=0.25, *args, **kwargs):
    # `clip_grad_norm` helps prevent the exploding gradient
    # problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


def get_state_dict_from_dp(pth_tar, state_key='state_dict'):
    # original saved file with DataParallel
    checkpoint = torch.load(pth_tar)
    state_dict = checkpoint[state_key]

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    # load params
    #model.load_state_dict(new_state_dict)
    
    return new_state_dict
