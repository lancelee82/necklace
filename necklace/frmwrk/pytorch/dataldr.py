""""""

# NOTE: now user should use torch.utils.data.distributed.DistributedSampler
#       when passing a dataloader_creator to distributed trainer, and that
#       make things more simpler

# TODO:
#    1. use distributed sampler
#    2. when dp+mp,... (using part of the world)

import torch


def get_dist_dataloader(dataset, *args, **kwargs):

    # NOTE: for distributed sampler/dataloader
    world_size = kwargs.get('world_size', 1)
    rank = kwargs.get('rank', 0)  # rank
    shuffle = kwargs.get('shuffle', True)

    num_workers = kwargs.get('num_workers', 8)
    pin_memory = kwargs.get('pin_memory', True)

    batch_size = kwargs.get('batch_size', 64)

    #kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    # ============================================================
    # NOTE: here we use DistributedSampler with necklace
    # ============================================================
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'sampler': train_sampler}

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        **kwargs)

    return data_loader

