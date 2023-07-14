"""NCCL Groups"""
#-----------------------------------------------------------------------------
# NOTE: from ncclgrp.py to all parallel training groups, see trnr/trnmode.py
#-----------------------------------------------------------------------------
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

import copy

#from .ncclwrp import pynccl
#from .ncclwrp import Nccl
#from .ncclwrp import NcclWrp


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

NCCL_GROUP_INIT_FLAG = False

NCCL_GROUP_MAIN = None
NCCL_GROUP_GRP_MAIN = []  # NOTE: this is not in NCCL_GROUPS_GRP_MAP
NCCL_GROUP_MAIN_WORLD_SIZE = None
NCCL_GROUP_MAIN_RANK = None

NCCL_MAIN_NK = None  # NOTE: one singleton Nccl() per one process

NCCL_GROUPS_MAP = {}
NCCL_GROUP_DP = None
NCCL_GROUP_MP = None
NCCL_GROUP_PP = None
NCCL_GROUP_ZR = None

NCCL_GROUPS_GRP_MAP = {}
NCCL_GROUP_GRP_DP = []
NCCL_GROUP_GRP_MP = []
NCCL_GROUP_GRP_PP = []
NCCL_GROUP_GRP_ZR = []


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


def get_nccl_group_main_grp_list():
    global NCCL_GROUP_GRP_MAIN
    return NCCL_GROUP_GRP_MAIN


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
def init_nccl_groups_map__bk(world_size, my_rank, mp_size, dp_size, is_worker):  # not used now
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


def get_nccl_group_pp(*args, **kwargs):
    global NCCL_GROUP_PP
    return NCCL_GROUP_PP


def get_nccl_group_zr(*args, **kwargs):
    global NCCL_GROUP_ZR
    return NCCL_GROUP_ZR


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


def get_nccl_group_pp_or_main(*args, **kwargs):
    global NCCL_GROUP_PP
    global NCCL_GROUP_MAIN
    if NCCL_GROUP_PP is not None:
        return NCCL_GROUP_PP
    else:
        return NCCL_GROUP_MAIN


def get_nccl_group_zr_or_main(*args, **kwargs):
    global NCCL_GROUP_ZR
    global NCCL_GROUP_MAIN
    if NCCL_GROUP_ZR is not None:
        return NCCL_GROUP_ZR
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


# ============================================================================
# pg group init api

def init_nccl_groups_map(tmode, cfg={}):
    global NCCL_GROUP_INIT_FLAG

    if not NCCL_GROUP_INIT_FLAG:
        init_nccl_pgs_map(tmode, cfg=cfg)

        print('NCCL_GROUPS_MAP', [k + ':' + str(v.to_cfg_dict()) for k, v in NCCL_GROUPS_MAP.items()])
        print('NCCL_GROUPS_GRP_MAP', NCCL_GROUPS_GRP_MAP)

    NCCL_GROUP_INIT_FLAG = True


def init_nccl_pgs_map(tmode, cfg={}):
    world_size = cfg.get('world_size')
    my_rank = cfg.get('my_rank')
    is_worker = cfg.get('is_worker')

    init_nccl_group_main(world_size, my_rank)

    pgs_dl, pgs_sz = build_nccl_pgs_map(tmode, cfg)
    #print('pgs_dl', pgs_dl)
    #print('pgs_sz', pgs_sz)

    init_nccl_group_pgs(world_size, my_rank, pgs_dl, pgs_sz, is_worker)


def init_nccl_group_main(world_size, my_rank):
    # Build the main group.
    global NCCL_GROUP_MAIN
    all_ranks = list(range(world_size))
    group = make_pg_by_nk(None, None, my_rank, all_ranks)
    NCCL_GROUP_MAIN = group
    set_nccl_group_main_world_size(world_size)
    set_nccl_group_main_rank(my_rank)

    #global NCCL_GROUPS_GRP_MAP
    #NCCL_GROUPS_GRP_MAP['main_group'] = [all_ranks,]
    global NCCL_GROUP_GRP_MAIN
    NCCL_GROUP_GRP_MAIN = [all_ranks,]


def init_nccl_group_pgs(world_size, my_rank, pgs_dl, pgs_sz, is_worker):
    global NCCL_GROUPS_GRP_MAP

    if len(pgs_dl) < 1:
        return  # if only one group, let it be the main

    for pg, grp in pgs_dl.items():
        for i, ranks in enumerate(grp):
            if is_worker:
                pg_sz = pgs_sz.get(pg)
                if my_rank in ranks:  # if my global rank in pg_grp
                    group = make_pg_by_nk(None, None, my_rank, ranks)
                    init_nccl_group_pg(pg, group, ranks)
        NCCL_GROUPS_GRP_MAP[pg] = grp


def init_nccl_group_pg(pg, group, ranks):

    global NCCL_GROUPS_MAP

    # Build the data parallel groups.
    global NCCL_GROUP_GRP_DP
    global NCCL_GROUP_DP
    global NCCL_GROUP_GRP_MP
    global NCCL_GROUP_MP
    global NCCL_GROUP_GRP_PP
    global NCCL_GROUP_PP
    global NCCL_GROUP_GRP_ZR
    global NCCL_GROUP_ZR

    # store the pg group in the map
    NCCL_GROUPS_MAP[pg] = group

    if pg == 'dp_group':
        assert NCCL_GROUP_DP is None, \
            'data parallel group is already initialized'
        NCCL_GROUP_DP = group
        NCCL_GROUP_GRP_DP = ranks

    elif pg == 'mp_group':
        assert NCCL_GROUP_MP is None, \
            'model parallel group is already initialized'
        NCCL_GROUP_MP = group
        NCCL_GROUP_GRP_MP = ranks

    elif pg == 'pp_group':
        assert NCCL_GROUP_PP is None, \
            'pipeline parallel group is already initialized'
        NCCL_GROUP_PP = group
        NCCL_GROUP_GRP_PP = ranks

    elif pg == 'zr_group':
        assert NCCL_GROUP_ZR is None, \
            'ZeRO parallel group is already initialized'
        NCCL_GROUP_ZR = group
        NCCL_GROUP_GRP_ZR = ranks

    else:
        pass  # raise


def build_nccl_pgs_map(tmode, cfg={}):
    world_size = cfg.get('world_size')
    my_rank = cfg.get('my_rank')
    dp_size = cfg.get('dp_size')
    pp_size = cfg.get('pp_size')
    mp_size = cfg.get('mp_size')
    zr_size = cfg.get('zr_size')

    if world_size is None or my_rank is None:
        raise Exception('Error: can NOT build nccl pgs map : (%s) [%s]' % (str(tmode), str(cfg)))

    if world_size < 1:
        raise Exception('Error: world_size(%d) should >1' % (world_size,))

    pgs = {}
    pgs_sz = {}

    if tmode == 'no':
        pass

    elif tmode in ['dp', 'pp', 'mp', 'zr']:
        all_ranks = list(range(world_size))
        k = tmode + '_group'
        pgs[k] = [all_ranks,]  # NOTE: one group in list
        pgs_sz[k] = world_size

    #elif tmode == 'dp+pp':
    #    pass  # TODO:

    elif tmode == 'dp+mp____':  # TODO: [===========================]
        assert mp_size > 1, 'mp_size should be >1'

        model_parallel_size = min(mp_size, world_size)
        ensure_divisibility(world_size, model_parallel_size)

        dp_grp = []
        for i in range(model_parallel_size):
            ranks = list(range(i, world_size, model_parallel_size))
            dp_grp.append(ranks)
        pgs['dp_group'] = dp_grp
        pgs_sz['dp_group'] = dp_size

        mp_grp = []
        for i in range(world_size // model_parallel_size):
            ranks = list(range(i * model_parallel_size,
                          (i + 1) * model_parallel_size))
            mp_grp.append(ranks)
            """
            if is_worker:
                if i == (rank // model_parallel_size):
                    group = make_pg_by_nk(None, None, rank, ranks)
                    NCCL_GROUP_MP = group
                    NCCL_GROUPS_MAP['mp_group'] = NCCL_GROUP_MP
            """
        pgs['mp_group'] = mp_grp
        pgs_sz['mp_group'] = mp_size


    #elif tmode == 'pp+mp':
    #    pass  # TODO:

    # --------------------------------------------------------------------
    # uniform for : aa+bb
    # --------------------------------------------------------------------
    elif tmode.count('*') == 1:  # not used for now
        aa, bb = tmode.split('*')
        aa_size = cfg.get(aa + '_size')
        bb_size = cfg.get(bb + '_size')

        assert bb_size > 1, 'bb_size should be >1'
        assert aa_size * bb_size == world_size, 'aa_size(%d) * bb_size(%d) != world_size(%d)' % (
            aa_size, bb_size, world_size)

        model_parallel_size = min(bb_size, world_size)
        ensure_divisibility(world_size, model_parallel_size)

        dp_grp = []
        for i in range(model_parallel_size):
            ranks = list(range(i, world_size, model_parallel_size))
            dp_grp.append(ranks)
        pgs[aa + '_group'] = dp_grp
        pgs_sz[aa + '_group'] = aa_size

        mp_grp = []
        for i in range(world_size // model_parallel_size):
            ranks = list(range(i * model_parallel_size,
                          (i + 1) * model_parallel_size))
            mp_grp.append(ranks)
        pgs[bb + '_group'] = mp_grp
        pgs_sz[bb + '_group'] = bb_size

    # --------------------------------------------------------------------
    # uniform for : aa+bb
    # --------------------------------------------------------------------
    elif tmode.count('+') == 1:  # NOTE: as a contrast
        aa, bb = tmode.split('+')
        aa_size = cfg.get(aa + '_size')
        bb_size = cfg.get(bb + '_size')

        assert bb_size > 1, 'bb_size should be >1'
        assert aa_size * bb_size == world_size, 'aa_size(%d) * bb_size(%d) != world_size(%d)' % (
            aa_size, bb_size, world_size)

        #bb_size = min(bb_size, world_size)
        #ensure_divisibility(world_size, bb_size)

        aa_grp, bb_grp = calc_pgs_p2(world_size, aa_size, bb_size)

        """
        aa_grp = []
        for i in range(world_size // aa_size):  # == bb_size
            ranks = list(range(i, world_size, bb_size))[:aa_size]
            aa_grp.append(ranks)

        bb_grp = []
        for i in range(world_size // bb_size):  # == aa_size
            #ranks = list(range(i * bb_size, (i + 1) * bb_size))
            ranks = list(range(i * bb_size, world_size, 1))[:bb_size]
            bb_grp.append(ranks)
        """

        pgs[aa + '_group'] = aa_grp
        pgs[bb + '_group'] = bb_grp
        pgs_sz[aa + '_group'] = aa_size
        pgs_sz[bb + '_group'] = bb_size


    elif tmode == 'dp+pp+mp____':
        pass

    elif tmode == 'dp+zr+mp____':
        pass

    elif tmode.count('+') == 2:  # NOTE:
        aa, bb, cc = tmode.split('+')
        aa_size = cfg.get(aa + '_size')
        bb_size = cfg.get(bb + '_size')
        cc_size = cfg.get(cc + '_size')

        assert bb_size > 1, 'bb_size should be >1'
        assert aa_size * bb_size * cc_size == world_size, '(%d)*(%d)*(%d) != world_size(%d)' % (
            aa_size, bb_size, cc_size, world_size)

        aa_grp, bb_grp, cc_grp = calc_pgs_p3(world_size, aa_size, bb_size, cc_size)

        pgs[aa + '_group'] = aa_grp
        pgs[bb + '_group'] = bb_grp
        pgs[cc + '_group'] = cc_grp
        pgs_sz[aa + '_group'] = aa_size
        pgs_sz[bb + '_group'] = bb_size
        pgs_sz[cc + '_group'] = cc_size


    elif tmode == 'dp+pp+zr+mp____':
        pass

    elif tmode.count('+') == 3:  # NOTE:
        aa, bb, cc, dd = tmode.split('+')
        aa_size = cfg.get(aa + '_size')
        bb_size = cfg.get(bb + '_size')
        cc_size = cfg.get(cc + '_size')
        dd_size = cfg.get(dd + '_size')

        assert bb_size > 1, 'bb_size should be >1'
        assert aa_size * bb_size * cc_size * dd_size == world_size, '(%d)*(%d)*(%d)*(%d) != world_size(%d)' % (
            aa_size, bb_size, cc_size, dd_size, world_size)

        aa_grp, bb_grp, cc_grp, dd_grp = calc_pgs_p4(world_size, aa_size, bb_size, cc_size, dd_size)

        pgs[aa + '_group'] = aa_grp
        pgs[bb + '_group'] = bb_grp
        pgs[cc + '_group'] = cc_grp
        pgs[dd + '_group'] = dd_grp
        pgs_sz[aa + '_group'] = aa_size
        pgs_sz[bb + '_group'] = bb_size
        pgs_sz[cc + '_group'] = cc_size
        pgs_sz[dd + '_group'] = dd_size


    return pgs, pgs_sz


def calc_pgs_p2(world_size, aa_size, bb_size):

    aa_grp = []
    for i in range(world_size // aa_size):  # == bb_size
        ranks = list(range(i, world_size, bb_size))[:aa_size]
        aa_grp.append(ranks)

    bb_grp = []
    for i in range(world_size // bb_size):  # == aa_size
        #ranks = list(range(i * bb_size, (i + 1) * bb_size))
        ranks = list(range(i * bb_size, world_size, 1))[:bb_size]
        bb_grp.append(ranks)

    return aa_grp, bb_grp


def calc_pgs_p3(world_size, aa_size, bb_size, cc_size):

    aa_grp = []
    for i in range(world_size // aa_size):
        ranks = list(range(i, world_size, (bb_size * cc_size)))[:aa_size]
        aa_grp.append(ranks)

    # -------------------------------------------------------------
    # split the whole world to parts of aa_size, and then calc
    # the pgs in the part worlds
    # -------------------------------------------------------------
    world_size_aa = world_size // aa_size

    bb_g, cc_g = calc_pgs_p2(world_size_aa, bb_size, cc_size)

    bb_grp = copy.copy(bb_g)
    cc_grp = copy.copy(cc_g)

    # -------------------------------------------------------------
    # copy the pgs in the part worlds by an offset, and then combine
    # them into the pgs of the whole world
    # -------------------------------------------------------------
    for j in range(1, aa_size):
        offset = j * (bb_size * cc_size)

        bb_go = []
        for bb in bb_g:
            bb_go.append([b + offset for b in bb])
        bb_grp += bb_go

        cc_go = []
        for cc in cc_g:
            cc_go.append([c + offset for c in cc])
        cc_grp += cc_go

    return aa_grp, bb_grp, cc_grp


def calc_pgs_p4(world_size, aa_size, bb_size, cc_size, dd_size):

    aa_grp = []
    for i in range(world_size // aa_size):
        ranks = list(range(i, world_size, (bb_size * cc_size * dd_size)))[:aa_size]
        aa_grp.append(ranks)

    # -------------------------------------------------------------
    # split the whole world to parts of aa_size, and then calc
    # the pgs in the part worlds
    # -------------------------------------------------------------
    world_size_aa = world_size // aa_size

    bb_g, cc_g, dd_g = calc_pgs_p3(world_size_aa, bb_size, cc_size, dd_size)

    bb_grp = copy.copy(bb_g)
    cc_grp = copy.copy(cc_g)
    dd_grp = copy.copy(dd_g)

    # -------------------------------------------------------------
    # copy the pgs in the part worlds by an offset, and then combine
    # them into the pgs of the whole world
    # -------------------------------------------------------------
    for j in range(1, aa_size):
        offset = j * (bb_size * cc_size * dd_size)

        bb_go = []
        for bb in bb_g:
            bb_go.append([b + offset for b in bb])
        bb_grp += bb_go

        cc_go = []
        for cc in cc_g:
            cc_go.append([c + offset for c in cc])
        cc_grp += cc_go

        dd_go = []
        for dd in dd_g:
            dd_go.append([d + offset for d in dd])
        dd_grp += dd_go

    return aa_grp, bb_grp, cc_grp, dd_grp

