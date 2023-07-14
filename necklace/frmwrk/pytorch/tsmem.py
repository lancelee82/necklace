"""Pytorch Tensor and Memory"""
#
# Memory Management for ZeRO
#
# NOTE:
#   1. mapping model weights / gradients to flatten 1d tersor (as memory)
#      =======================================================================
#      (if using 1d_mem, we can rebuid the compute graph by reloading weigths
#       and saved activations to do backward and obtain the gradients for opt)
#      =======================================================================
#
# TEST: 
#   1. tests/test_pt_flat_weig_grad.py
#
# TODO:
#   1. if not p.requires_grad or p.grad is None:
#
# ----------------------------------------------------------------------------

import time

import torch 
import torch.nn as nn
import torch.nn.functional as F


# =================================================================
# flatten by new 1d tersor

def md_params_to_flatten_1d_new(md, device):
    total_size = 0
    for nm, p in md.named_parameters(recurse=True):
        #
        #if not p.requires_grad or p.grad is None:
        #print(nm, p.data.numel())
        total_size += p.data.numel()

    #print('total_size', total_size)

    prm_fltn = torch.empty(total_size, dtype=p.data.dtype, device=device)

    fltn_start = 0
    for nm, p in md.named_parameters(recurse=True):
        #if not p.requires_grad or p.grad is None:
        p_len = p.data.numel()
        #print(fltn_start, '-=>', p_len)
        dst = torch.narrow(prm_fltn, 0, fltn_start, p_len)
        dst.copy_(p.data.view(-1))
        p.data = dst.view_as(p.data)
        fltn_start += p_len

    return prm_fltn


def md_grads_to_flatten_1d_new(md, device):
    total_size = 0
    for nm, p in md.named_parameters(recurse=True):
        #if not p.requires_grad or p.grad is None:
        #print(nm, p.grad.data.numel())
        total_size += p.grad.data.numel()
        # TODO: p.grad may be None before the first backward

    #print('total_size', total_size)

    grd_fltn = torch.empty(total_size, dtype=p.grad.data.dtype, device=device)

    grd_fltn.zero_()

    fltn_start = 0
    for nm, p in md.named_parameters(recurse=True):
        #if not p.requires_grad or p.grad is None:
        p_len = p.grad.data.numel()
        #print(fltn_start, '-=>', p_len)
        dst = torch.narrow(grd_fltn, 0, fltn_start, p_len)
        dst.copy_(p.grad.data.view(-1))
        p.grad.data = dst.view_as(p.grad.data)
        fltn_start += p_len

    return grd_fltn


# =================================================================
# flatten by existed 1d tersor

def md_flatten_1d_mem_create(total_size, dtype=torch.float, device=torch.device('cpu')):
    fltn = torch.empty(total_size, dtype=dtype, device=device)
    fltn.zero_()
    return fltn


def md_flatten_1d_mem_reset(fltn):
    fltn.zero_()


def md_flatten_1d_mem_copy(frm_fltn, to_fltn, sz=-1):
    to_fltn.copy_(frm_fltn)


def md_params_numel_total_size(md):
    total_size = 0
    for nm, p in md.named_parameters(recurse=True):
        #
        #if not p.requires_grad or p.grad is None:
        #print(nm, p.data.numel())
        total_size += p.data.numel()

    #print('total_size', total_size)

    return total_size


def md_params_to_flatten_1d_mem(md, prm_fltn):
    fltn_start = 0
    for nm, p in md.named_parameters(recurse=True):
        #if not p.requires_grad or p.grad is None:
        p_len = p.data.numel()
        #print(fltn_start, '-=>', p_len)
        dst = torch.narrow(prm_fltn, 0, fltn_start, p_len)
        dst.copy_(p.data.view(-1))  # TODO: --> .storage().set_(...)
        p.data = dst.view_as(p.data)
        fltn_start += p_len

    return prm_fltn


def md_grads_numel_total_size(md):
    total_size = 0
    for nm, p in md.named_parameters(recurse=True):
        #if not p.requires_grad or p.grad is None:
        #print(nm, p.grad.data.numel())
        total_size += p.grad.data.numel()

    #print('total_size', total_size)

    return total_size


def md_grads_to_flatten_1d_mem(md, grd_fltn):
    fltn_start = 0
    for nm, p in md.named_parameters(recurse=True):
        #if not p.requires_grad or p.grad is None:
        if p.grad is None:  # at the init there is no grad in p
            continue
        p_len = p.grad.data.numel()
        #print(fltn_start, '-=>', p_len)
        dst = torch.narrow(grd_fltn, 0, fltn_start, p_len)
        dst.copy_(p.grad.data.view(-1))
        p.grad.data = dst.view_as(p.grad.data)
        fltn_start += p_len

    return grd_fltn


# =================================================================
# memory manager

class PtCudaFltnMemManager(object):
    def __init__(self, rank=0, ctx=None, max_weig_total_size=0, max_grad_total_size=0):
        self.rank = rank
        self.ctx = ctx
        self.max_weig_total_size = max_weig_total_size
        self.max_grad_total_size = max_grad_total_size

        self.cpu = torch.device('cpu')

        if self.max_weig_total_size > 0:
            self.prm_fltn_gpu = md_flatten_1d_mem_create(
                self.max_weig_total_size,
                dtype=torch.float,
                device=self.ctx
            )
        else:
            self.prm_fltn_gpu = None

        if self.max_grad_total_size > 0:
            self.grd_fltn_gpu = md_flatten_1d_mem_create(
                self.max_grad_total_size,
                dtype=torch.float,
                device=self.ctx
            )
        else:
            self.grd_fltn_gpu = None

        self.prm_fltns_cpu = {}
        self.grd_fltns_cpu = {}

    def md_to_cpu(self, md):
        # NOTE: the md is a mdwrp.MdWrpWithSelfBkwd

        if md.mp_order not in self.prm_fltns_cpu.keys():
            prm_fltn_cpu = md_params_to_flatten_1d_new(md, self.cpu)
            self.prm_fltns_cpu[md.mp_order] = prm_fltn_cpu
        else:
            prm_fltn_cpu = self.prm_fltns_cpu[md.mp_order]
            md_params_to_flatten_1d_mem(md, prm_fltn_cpu)

        if md.mp_order not in self.grd_fltns_cpu.keys():
            #grd_fltn_cpu = md_grads_to_flatten_1d_new(md, self.cpu)  # NOTE: maybe p.grad is None
            grd_fltn_cpu = md_flatten_1d_mem_create(
                self.max_grad_total_size,
                dtype=torch.float,
                device=self.cpu
            )
            self.grd_fltns_cpu[md.mp_order] = grd_fltn_cpu
            md_grads_to_flatten_1d_mem(md, grd_fltn_cpu)
        else:
            grd_fltn_cpu = self.grd_fltns_cpu[md.mp_order]
            md_grads_to_flatten_1d_mem(md, grd_fltn_cpu)

    def md_to_gpu(self, md):
        md_params_to_flatten_1d_mem(md, self.prm_fltn_gpu)
        md_grads_to_flatten_1d_mem(md, self.grd_fltn_gpu)

    def zero_grad(self):
        md_flatten_1d_mem_reset(self.grd_fltn_gpu)
        for i, grd_fltn_cpu in self.grd_fltns_cpu.items():
            md_flatten_1d_mem_reset(grd_fltn_cpu)
