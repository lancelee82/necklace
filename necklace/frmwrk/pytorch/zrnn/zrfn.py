"""ZeRO Function"""

import torch

from necklace.cuda import ncclgrp
from necklace.cuda import ncclfns

from .utils import split_tensor_along_last_dim


# ------------------------------------------
# ZeRO ProcessGroup functions.
# ------------------------------------------

def zr_pg_all_reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = ncclgrp.get_nccl_group_zr_or_main()

    # Bypass the function if we are using only 1 GPU.
    if ncclgrp.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    ncclfns.all_reduce(input_, group=group)

    #print('mappings.zr_pg_all_reduce:', group.to_cfg_dict())

    return input_


def zr_pg_split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = ncclgrp.get_nccl_group_zr_or_main()

    # Bypass the function if we are using only 1 GPU.
    if ncclgrp.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = ncclgrp.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = ncclgrp.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def zr_pg_all_gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = ncclgrp.get_nccl_group_zr_or_main()

    # Bypass the function if we are using only 1 GPU.
    if ncclgrp.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = ncclgrp.get_rank(group=group)
    world_size = ncclgrp.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    ncclfns.all_gather(output, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    #output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def zr_pg_bcast(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = ncclgrp.get_nccl_group_zr_or_main()

    # Bypass the function if we are using only 1 GPU.
    if ncclgrp.get_world_size(group=group) == 1:
        return input_

    # Broadcast.
    ncclfns.broadcast(input_, group=group)  # TODO: root=

    #print('mappings.zr_pg_bcast:', group.to_cfg_dict())

    return input_


def zr_pg_reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = ncclgrp.get_nccl_group_zr_or_main()

    # Bypass the function if we are using only 1 GPU.
    if ncclgrp.get_world_size(group=group) == 1:
        return input_

    # Reduce.
    ncclfns.reduce(input_, group=group)  # TODO: root=

    #print('mappings.zr_pg_reduce:', group.to_cfg_dict())

    return input_


# ------------------------------------------
# Autograd Functions.
# ------------------------------------------

class _BcastToZeROParallelRegion(torch.autograd.Function):
    """Bcast the weights to the zero parallel region."""

    @staticmethod
    def forward(ctx, weights_):
        return zr_pg_bcast(weights_)
        # TODO: activations checkpointing [================================]

    @staticmethod
    def backward(ctx, grad_output):
        return zr_pg_reduce(grad_output)


# ------------------------------------------
# Helper functions.
# ------------------------------------------

def bcast_to_zero_parallel_region(weights_):
    return _BcastToZeROParallelRegion.apply(weights_)
