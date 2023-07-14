"""NCCL Groups"""
#
# ref: /pytorch/torch/distributed/distributed_c10d.py
#
# NOTE: 1. the part of 'Model and Data parallel groups' at:
#          /necklace/frmwrk/pytorch/mpnn/initialize.py
#
# TODO: 1. main pg, dp pg, mp pg [o]
#       2. other groups (pipeline parallel group ?)
#       3. p2p groups (maybe should use main group ?)
#
#-----------------------------------------------------------------------------

from .ncclwrp import pynccl
from .ncclwrp import Nccl
from .ncclwrp import NcclWrp


class NcclProcessGroup(object):
    def __init__(self, nc, nk, comm_i, pg_rank, pg_ranks, *args, **kwargs):
        self.nc = nc
        if nk is None and nc is not None:
            self.nk = nc.nk
        else:
            self.nk = nk  # the global nccl API
        self.comm_i = comm_i  # myself nccl comm
        self.pg_rank = pg_rank  # my global rank
        self.pg_ranks = sorted(pg_ranks)

        self.rank = self.pg_ranks.index(self.pg_rank)
        self.ranks = list(range(len(self.pg_ranks)))
        self.world_size = len(self.pg_ranks)

        #self.ranks_map_to
        #self.ranks_map_from

    def to_cfg_dict(self):
        cfg = {
            'pg_rank': self.pg_rank,
            'pg_ranks': self.pg_ranks,
            'rank': self.rank,
            'ranks': self.ranks,
            'world_size': self.world_size,
        }
        return cfg

    def set_comm_i(self, comm_i):
        self.comm_i = comm_i

    def create_comm_i(self):
        raise NotImplemented('create_comm_i: call zrpc')

    def stream_sync(self):  # from pynccl api
        stream_i = self.nk.get_stream()
        stream_i.synchronize()


def build_pg_cfg(pg_rank, pg_ranks):
        pg_ranks = sorted(pg_ranks)
        rank = pg_ranks.index(pg_rank)
        ranks = list(range(len(pg_ranks)))
        world_size = len(pg_ranks)

        cfg = {
            'nk': None,
            'comm_i': None,

            'pg_rank': pg_rank,
            'pg_ranks': pg_ranks,
            'rank': rank,
            'ranks': ranks,
            'world_size': world_size,
        }

        return cfg


def make_pg_by_nc(nc, pg_rank, pg_ranks):
    pg = NcclProcessGroup(
        nc, nc.nk, nc.comm_i,
        pg_rank, pg_ranks
    )
    return pg


def make_pg_by_nk(nk, comm_i, pg_rank, pg_ranks):
    pg = NcclProcessGroup(
        None, nk, comm_i,
        pg_rank, pg_ranks
    )
    return pg


# ============================================================================
# main groups and api

NCCL_GROUP_MAIN = None
NCCL_GROUP_MAIN_WORLD_SIZE = None
NCCL_GROUP_MAIN_RANK = None

NCCL_MAIN_NK = None  # NOTE: one singleton Nccl() per one process

NCCL_GROUPS_MAP = {}
NCCL_GROUP_DP = None
NCCL_GROUP_MP = None

NCCL_GROUPS_GRP_MAP = {}
NCCL_GROUP_GRP_DP = []
NCCL_GROUP_GRP_MP = []


# ============================================================================
# NOTE  1. About the main group which contains all the nodes, is created
#          at init_nccl_groups_map(), and is updated after finishing the main
#          rpc init_nccl_comm() in trainer/tnopbs.py by calling this func
# ============================================================================
def set_nccl_group_main_by_nc(nc, world_size):
    global NCCL_GROUP_MAIN

    if NCCL_GROUP_MAIN is None:  # if not call init_nccl_groups_map()
        pg_ranks = list(range(world_size))
        pg = make_pg_by_nc(nc, nc.rank, pg_ranks)
        NCCL_GROUP_MAIN = pg

    NCCL_GROUP_MAIN.nc = nc
    NCCL_GROUP_MAIN.nk = nc.nk
    NCCL_GROUP_MAIN.comm_i = nc.comm_i

    global NCCL_MAIN_NK
    NCCL_MAIN_NK = nc.nk


def get_nccl_group_main(*args, **kwargs):
    global NCCL_GROUP_MAIN
    return NCCL_GROUP_MAIN


def set_nccl_group_main_world_size(world_size):
    global NCCL_GROUP_MAIN_WORLD_SIZE
    NCCL_GROUP_MAIN_WORLD_SIZE = world_size


def get_nccl_group_main_world_size(*args, **kwargs):
    global NCCL_GROUP_MAIN_WORLD_SIZE
    return NCCL_GROUP_MAIN_WORLD_SIZE


def set_nccl_group_main_rank(rank):
    global NCCL_GROUP_MAIN_RANK
    NCCL_GROUP_MAIN_RANK = rank


def get_nccl_group_main_rank(*args, **kwargs):
    global NCCL_GROUP_MAIN_RANK
    return NCCL_GROUP_MAIN_RANK


def get_nccl_main_nk(*args, **kwargs):
    global NCCL_MAIN_NK
    return NCCL_MAIN_NK


# ============================================================================
# dp / mp groups and api

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# ============================================================================
# NOTE: 1. After calling this function, the nk and comm_i are None and will be
#          set a moment later
#       2. About the nk (pynccl.Nccl) in the groups (main and dp and mp),
#          only one nk is used at all, which is same with the main_pg and is set
#          after finishing the init_nccl_comm_pg() in trainer/tnopbs.py
#       3. About the comm_i (nccl communicator) in the groups (dp and mp),
#          every group has its own comm_i with different world_size and ranks,
#          and is set after finishing the init_nccl_comm_pg() in trainer/tnopbs.py
# ============================================================================
def init_nccl_groups_map(world_size, my_rank, mp_size, dp_size, is_worker):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel grous as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if my_rank == 0:
        print('> initializing model parallel with size {}'.format(mp_size))
    # Get world size and rank. Ensure some consistencies.
    model_parallel_size = min(mp_size, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    rank = my_rank

    # Build the main group.
    global NCCL_GROUP_MAIN
    all_ranks = list(range(world_size))
    group = make_pg_by_nk(None, None, rank, all_ranks)
    NCCL_GROUP_MAIN = group
    set_nccl_group_main_world_size(world_size)
    set_nccl_group_main_rank(rank)

    global NCCL_GROUPS_MAP
    global NCCL_GROUPS_GRP_MAP

    # Build the data parallel groups.
    global NCCL_GROUP_GRP_DP
    global NCCL_GROUP_DP
    assert NCCL_GROUP_DP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = list(range(i, world_size, model_parallel_size))
        NCCL_GROUP_GRP_DP.append(ranks)
        if is_worker:
            if i == (rank % model_parallel_size):
                group = make_pg_by_nk(None, None, rank, ranks)
                NCCL_GROUP_DP = group
                NCCL_GROUPS_MAP['dp_group'] = NCCL_GROUP_DP
    NCCL_GROUPS_GRP_MAP['dp_group'] = NCCL_GROUP_GRP_DP

    # Build the model parallel groups.
    global NCCL_GROUP_GRP_MP
    global NCCL_GROUP_MP
    assert NCCL_GROUP_MP is None, \
        'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = list(range(i * model_parallel_size,
                      (i + 1) * model_parallel_size))
        NCCL_GROUP_GRP_MP.append(ranks)
        if is_worker:
            if i == (rank // model_parallel_size):
                group = make_pg_by_nk(None, None, rank, ranks)
                NCCL_GROUP_MP = group
                NCCL_GROUPS_MAP['mp_group'] = NCCL_GROUP_MP
    NCCL_GROUPS_GRP_MAP['mp_group'] = NCCL_GROUP_GRP_MP

    print('NCCL_GROUPS_MAP', [v.to_cfg_dict() for k, v in NCCL_GROUPS_MAP.items()])
    print('NCCL_GROUPS_GRP_MAP', NCCL_GROUPS_GRP_MAP)


def set_nccl_groups_map(pg_map):
    global NCCL_GROUPS_MAP
    NCCL_GROUPS_MAP = pg_map


def get_nccl_groups_map():
    global NCCL_GROUPS_MAP
    return NCCL_GROUPS_MAP


def get_nccl_groups_map_cfg_dict():
    cfg_map = {}

    global NCCL_GROUPS_MAP
    for k, v in NCCL_GROUPS_MAP.items():
        cfg_map[k] = v.to_cfg_dict()

    return cfg_map


def get_nccl_groups_map_grp_dict():
    global NCCL_GROUPS_GRP_MAP
    return NCCL_GROUPS_GRP_MAP


def get_nccl_group_dp(*args, **kwargs):
    global NCCL_GROUP_DP
    return NCCL_GROUP_DP


def get_nccl_group_mp(*args, **kwargs):
    global NCCL_GROUP_MP
    return NCCL_GROUP_MP


def get_nccl_group_dp_or_main(*args, **kwargs):
    global NCCL_GROUP_DP
    global NCCL_GROUP_MAIN
    if NCCL_GROUP_DP is not None:
        return NCCL_GROUP_DP
    else:
        return NCCL_GROUP_MAIN


def get_nccl_group_mp_or_main(*args, **kwargs):
    global NCCL_GROUP_MP
    global NCCL_GROUP_MAIN
    if NCCL_GROUP_MP is not None:
        return NCCL_GROUP_MP
    else:
        return NCCL_GROUP_MAIN


def del_nccl_groups_map_comms():
    global NCCL_MAIN_NK
    global NCCL_GROUPS_MAP
    for k, v in NCCL_GROUPS_MAP.items():
        NCCL_MAIN_NK.comm_destroy(v.comm_i)


# ============================================================================
# unified api

def get_rank(group=None):
    if group is None:
        group = NCCL_GROUP_MAIN
    if group is None:
        return NCCL_GROUP_MAIN_RANK
    return group.rank


def get_world_size(group=None):
    if group is None:
        group = NCCL_GROUP_MAIN
    if group is None:
        return NCCL_GROUP_MAIN_WORLD_SIZE
    return group.world_size
