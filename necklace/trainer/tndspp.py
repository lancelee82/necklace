"""Distributed Model-Parallel (horizontally splitted) Trainer Services (Scheduler + Worker)"""
from __future__ import print_function, absolute_import, division

import time
import uuid
from collections import OrderedDict

from necklace.rpc import zrpc
from necklace.rpc import svctmp
from necklace.rpc import msgpck
from necklace.rpc import ilog


ILOG_I_MP_SVR_WKR = 600011
ILOG_I_MP_SVR_WKR_PRT = 600015
ILOG_I_MP_SVC_WKR = 600012
ILOG_I_MP_SVC_WKR_PRT = 600016
ILOG_I_MP_SVR_SCH = 600021
ILOG_I_MP_SVR_SCH_PRT = 600025
ILOG_I_MP_SVC_SCH = 600022
ILOG_I_MP_SVC_SCH_PRT = 600026

Q_STATE_HEAD_MP_WKR = 610000
Q_STATE_HEAD_MP_SCH = 620000


class MPSVRWorker(svctmp.TmplServer):
    def __init__(self, cfg={}):
        super(MPSVRWorker, self).__init__(cfg)

        self.state = 0

        self.trainer_cls = cfg.get('trainer_cls')
        if not self.trainer_cls:
            raise Exception('No trainer_cls in cfg')

        self.trainer_cfg = cfg.get('trainer_cfg')
        if not self.trainer_cfg:
            raise Exception('No trainer_cfg in cfg')

        self.trnr = self.trainer_cls(self.trainer_cfg)

        self.scheduler_url = self.cfg.get('scheduler_url')
        self.scheduler_cli = self.mk_rpc_cli(self.scheduler_url)

    def do_worker_reg(self, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_worker_reg')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_worker_reg',
            url=self.url)
        return r

    # ----------------- worker_reg -----------------

    def fn_init_train(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_init_train')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(1, msg)

        rep = self.mk_rpc_ret('fn_init_train', 'ok')
        return self.msg_pck(rep)

    def fn_cre_nccl(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_cre_nccl')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(2, msg)

        rep = self.mk_rpc_ret('fn_cre_nccl', 'ok')
        return self.msg_pck(rep)

    def fn_cre_nccl_nuid(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_cre_nccl_nuid')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        #self.q_put_state(3, msg)
        nuid = self.trnr.cre_nccl_nuid(msg)
        nuid = msgpck.pkl_dumps(nuid)

        rep = self.mk_rpc_ret('fn_cre_nccl_nuid', 'ok', nuid=nuid)
        return self.msg_pck(rep)

    def fn_init_nccl_comm(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_init_nccl_comm')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(4, msg)

        rep = self.mk_rpc_ret('fn_init_nccl_comm', 'ok')
        return self.msg_pck(rep)

    # ----------------- init_nccl -----------------

    # ============================================================================== from: tndsms
    def fn_cre_nccl_nuid_pg(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_cre_nccl_nuid_pg')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        #self.q_put_state(3, msg)
        nuid = self.trnr.cre_nccl_nuid_pg(msg)
        nuid = msgpck.pkl_dumps(nuid)

        rep = self.mk_rpc_ret('fn_cre_nccl_nuid_pg', 'ok', nuid=nuid)
        return self.msg_pck(rep)

    def fn_init_nccl_comm_pg(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_init_nccl_comm_pg')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(900004, msg)

        rep = self.mk_rpc_ret('fn_init_nccl_comm_pg', 'ok')
        return self.msg_pck(rep)
    # ============================================================================== from: tndsms

    # ----------------- init_nccl pgs -----------------




    # ============================================================================== from: tndsms
    def fn_one_step_start(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_step_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(11, msg)
        #r = self.trnr.one_step()
        #r = self.trnr.check_msg(msg)

        rep = self.mk_rpc_ret('fn_one_step_start', 'ok')
        return self.msg_pck(rep)

    def fn_nccl_all_reduce_start(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_nccl_all_reduce_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(12, msg)
        #r = self.trnr.do_params_nccl_all_reduce()
        #r = self.trnr.check_msg(msg)

        rep = self.mk_rpc_ret('fn_nccl_all_reduce_start', 'ok')
        return self.msg_pck(rep)

    def nccl_stream_sync(self, ret, *args, **kwargs):
        self.trnr.nccl_stream_sync()

    def do_one_step_end(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_step_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_step_end',
            ret=ret)
        return r

    def do_nccl_all_reduce_end(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_nccl_all_reduce_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_nccl_all_reduce_end',
            ret=ret)
        return r
    # ============================================================================== from: tndsms




    def fn_next_epoch_start(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_next_epoch_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(13, msg)

        rep = self.mk_rpc_ret('fn_next_epoch_start', 'ok')
        return self.msg_pck(rep)

    def do_next_epoch_end(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_next_epoch_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_next_epoch_end',
            ret=ret)
        return r

    # ----------------- epoch_func -----------------

    def fn_stop_train(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_stop_train')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(20, msg)
        #self.trainer_stop()

        rep = self.mk_rpc_ret('fn_stop_train', 'ok')
        return self.msg_pck(rep)

    def trainer_stop(self):
        r = self.trnr.clear()

    # ----------------- train_func -----------------

    def fn_one_mp_step_start(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_mp_step_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(Q_STATE_HEAD_MP_WKR + 201, msg)

        rep = self.mk_rpc_ret('fn_one_mp_step_start', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_ack(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_ack')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_ack',
            ret=ret)
        return r

    def do_one_mp_step_end(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_end',
            ret=ret)
        return r

    def fn_one_mp_step_forward(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_mp_step_forward')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(Q_STATE_HEAD_MP_WKR + 203, msg)

        rep = self.mk_rpc_ret('fn_one_mp_step_forward', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_forwarded(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_forwarded')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_forwarded',
            ret=ret)
        return r

    def fn_one_mp_step_trans_outputs(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_mp_step_trans_outputs')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(Q_STATE_HEAD_MP_WKR + 205, msg)

        rep = self.mk_rpc_ret('fn_one_mp_step_trans_outputs', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_outputs_transed(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_outputs_transed')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_outputs_transed',
            ret=ret)
        return r

    def fn_one_mp_step_calc_loss(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_mp_step_calc_loss')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(Q_STATE_HEAD_MP_WKR + 221, msg)

        rep = self.mk_rpc_ret('fn_one_mp_step_calc_loss', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_loss_calced(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_loss_calced')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_loss_calced',
            ret=ret)
        return r

    def fn_one_mp_step_trans_grads(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_mp_step_trans_grads')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(Q_STATE_HEAD_MP_WKR + 223, msg)

        rep = self.mk_rpc_ret('fn_one_mp_step_trans_grads', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_grads_transed(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_grads_transed')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_grads_transed',
            ret=ret)
        return r

    def fn_one_mp_step_backward(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> fn_one_mp_step_backward')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(Q_STATE_HEAD_MP_WKR + 225, msg)

        rep = self.mk_rpc_ret('fn_one_mp_step_backward', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_backwarded(self, ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_WKR, '>>> do_one_mp_step_backwarded')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_mp_step_backwarded',
            ret=ret)
        return r

    # ----------------- one_mp_step -----------------


# ####################################################################################################


class MPSVCWorker(svctmp.TmplService):
    def __init__(self, cfg={}):
        super(MPSVCWorker, self).__init__(cfg)

        self.init_svr()

        self.reg_worker()  # global entry

    def init_svr(self):
        self.cfg['q'] = self.q
        svr = MPSVRWorker(self.cfg)
        self.svr = svr

    def reg_worker(self):
        self.q_put_state(40, 'worker_reg')

    def check_state(self, *args, **kwargs):

        s = self.q_get_state(timeout=1)
        svr_state, sa, skw = s
        ilog.debug(ILOG_I_MP_SVC_WKR, '=== svr_state', self.svr.rank, svr_state)

        # NOTE: wait the zrpc call to return
        self.g_sleep(0.0001)

        if svr_state == 0:
            pass

        elif svr_state == 40:  # worker_reg
            self.svr.do_worker_reg()

        # ----------------- worker_reg -----------------

        elif svr_state == 1:  # init_train
            pass

        elif svr_state == 2:  # fn_cre_nccl
            msg = sa[0]
            cfg = msg.get('cfg')
            r = self.svr.trnr.cre_nccl(msg)
            #self.q_put_state(0, r)

        elif svr_state == 3:  # cre_nccl_nuid
            pass

        elif svr_state == 4:
            msg = sa[0]
            nuid = msg.get('nuid')
            nuid = msgpck.pkl_loads(nuid)
            msg['nuid'] = nuid

            # NOTE: wait the zrpc call to return
            self.g_sleep(1.0)

            r = self.svr.trnr.init_nccl_comm(msg)  # nuid
            #self.q_put_state(0, r)

        # ----------------- init_nccl -----------------


        # ============================================================================== from: tndsms
        elif svr_state == 900004:
            msg = sa[0]
            cfg = msg.get('cfg')
            nuid = cfg.get('nuid')
            nuid = msgpck.pkl_loads(nuid)
            cfg['nuid'] = nuid

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.1)

            r = self.svr.trnr.init_nccl_comm_pg(cfg)  # nuid
            #self.q_put_state(0, r)

        # ----------------- init_nccl pgs -----------------

        elif svr_state == 11:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.0001)

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_step_end(ret)
            #self.q_put_state(0, r)

        elif svr_state == 12:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.0001)

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)

            # TODO: no need to call nccl_stream_sync here [?]
            self.q_put_state(42, ret)
            # or,
            #r = self.svr.do_nccl_all_reduce_end(ret)
            #self.q_put_state(0, r)

        elif svr_state == 42:
            ret = sa[0]
            r = self.svr.nccl_stream_sync(ret)
            r = self.svr.do_nccl_all_reduce_end(ret)

        # ============================================================================== from: tndsms



        elif svr_state == 13:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.01)

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_next_epoch_end(ret)
            #self.q_put_state(0, r)

        elif svr_state == 20:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.01)

            r = self.svr.trainer_stop()
            self.stop()  # service loop stop

        # ----------------- epoch_func -----------------

        elif svr_state == Q_STATE_HEAD_MP_WKR + 201:

            # NOTE: wait the zrpc call to return
            #self.g_sleep(0.0001)

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_mp_step_ack(ret)

        elif svr_state == Q_STATE_HEAD_MP_WKR + 203:

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_mp_step_forwarded(ret)

        elif svr_state == Q_STATE_HEAD_MP_WKR + 205:

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_mp_step_outputs_transed(ret)

        elif svr_state == Q_STATE_HEAD_MP_WKR + 221:

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_mp_step_loss_calced(ret)

        elif svr_state == Q_STATE_HEAD_MP_WKR + 223:

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_mp_step_grads_transed(ret)

        elif svr_state == Q_STATE_HEAD_MP_WKR + 225:

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_one_mp_step_backwarded(ret)



# ####################################################################################################
# ####################################################################################################
# ####################################################################################################



class MPSVRScheduler(svctmp.TmplServer):
    def __init__(self, cfg={}):
        super(MPSVRScheduler, self).__init__(cfg)

        self.state = 0

        self.stopped = 0

        self.epochs = self.cfg.get('epochs', 1)
        self.epoch = 0
        self.epoch_end_flags = {}

        self.weights_sync_flags = {}
        self.dt_sampler_sync_flags = {}
        self.one_step_flags = {}
        self.nccl_all_reduce_flags = {}
        self.next_epoch_flags = {}

        self.net_map = self.cfg.get('net_map')
        if not self.net_map:
            raise Exception('no net_map in scheduler')  # TODO:

        self.one_mp_step_flags = {}
        self.one_mp_step_order_i = 0
        self.one_mp_step_order_n = len(self.net_map.keys())  # NOTE: use self.pp_group_l

        self.mp_target_shps = None

        self.one_mp_step_forward_flags = {}
        self.one_mp_step_trans_outputs_flags = {}
        self.one_mp_step_calc_loss_flags = {}
        self.one_mp_step_trans_grads_flags = {}
        self.one_mp_step_backward_flags = {}

        self.mp_epoch_end_flags = {}

        #self.worker_urls = self.cfg.get('worker_urls')
        #self.init_worker_clis(self.worker_urls)
        # --> wait for worker to register themself

        #self.wdsz = self.cfg.get('wdsz')

        self.workers = self.cfg.get('workers')
        self.worker_eps = OrderedDict()

        # ============================================================================== from: tndsms
        self.nccl_groups_cfg_map = self.cfg.get('nccl_groups_cfg_map', {})
        self.nccl_groups_grp_map = self.cfg.get('nccl_groups_grp_map', {})
        self.nccl_groups_grp_list = []
        for k, v in self.nccl_groups_grp_map.items():
            self.nccl_groups_grp_list.append([k, v])
        self.nccl_groups_grp_cnt = 0
        self.nccl_groups_nuid_cache = {}

        # ============================================================================== add: for dp+mp
        self.nccl_group_main_grp_list = self.cfg.get('nccl_group_main_grp_list', [])
        self.pp_group = self.nccl_groups_grp_map.get('pp_group', [])
        if not self.pp_group:
            self.pp_group = self.nccl_group_main_grp_list
        self.pp_group_n = len(self.pp_group)
        self.pp_group_l = len(self.pp_group[0])
        print('pp_group: ', self.pp_group)

    def reset_mp(self):
        self.one_mp_step_order_i = 0
        self.mp_target_shps = None

    def init_worker_clis(self, worker_urls):
        self.worker_clis = OrderedDict()
        for wkr in worker_urls:
            cli = self.mk_rpc_cli(wkr)
            self.worker_clis[wkr] = cli

        self.kn = len(worker_urls)  # == self.workers

    def rank_key(self, rank):
        return 'rank-' + str(rank)

    def worker_url(self, rank):
        k = self.rank_key(rank)
        wkr_url = self.worker_eps[k]
        return wkr_url

    def worker_cli(self, rank):
        url = self.worker_url(rank)
        cli = self.worker_clis[url]
        return cli

    def check_workers_reg(self, rank=None, role=None, msg=None, *args, **kwargs):
        if not msg:
            return False

        wkr_url = msg.get('url')

        k = self.rank_key(rank)
        self.worker_eps[k] = wkr_url

        if len(self.worker_eps.keys()) >= self.workers:
            self.init_worker_clis(self.worker_eps.values())
            return True
        else:
            return False

    def fn_worker_reg(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_worker_reg')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(41, msg, rank, role)

        rep = self.mk_rpc_ret('fn_worker_reg', 'ok')
        return self.msg_pck(rep)

    def do_rpc_call_wkrs(self, fn, *args, **kwargs):
        for wkr, cli in self.worker_clis.items():
            r = self.do_rpc_call(
                cli, fn, *args, **kwargs)

            # TODO: check return

        return r

    def do_rpc_call_wkr(self, wkr_rank, fn, *args, **kwargs):
        cli = self.worker_cli(wkr_rank)
        r = self.do_rpc_call(
            cli, fn, *args, **kwargs)

        # TODO: check return

        return r

    def do_rpc_call_some_wkrs(self, wkr_ranks, fn, *args, **kwargs):
        for wkr_rank in wkr_ranks:
            cli = self.worker_cli(wkr_rank)
            r = self.do_rpc_call(
                cli, fn, *args, **kwargs)

            # TODO: check return

        return r

    def do_rpc_call_wkrs_p2p(self, from_rank, to_rank, fn, *args, **kwargs):
        from_cli = self.worker_cli(from_rank)
        r = self.do_rpc_call(
            from_cli, fn, *args, from_rank=from_rank, to_rank=to_rank, **kwargs)

        to_cli = self.worker_cli(to_rank)
        r = self.do_rpc_call(
            to_cli, fn, *args, from_rank=from_rank, to_rank=to_rank, **kwargs)

        # TODO: check return

        return r

    def do_rpc_call_some_wkrs_p2p(self, from_ranks, to_ranks, fn, *args, **kwargs):
        for from_rank, to_rank in zip(from_ranks, to_ranks):
            r = self.do_rpc_call_wkrs_p2p(from_rank, to_rank, fn, *args, **kwargs)
        return r


    # ----------------- worker_reg -----------------

    def do_init_train(self, init_cfg={}, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_init_train')
        r = self.do_rpc_call_wkrs(
            'fn_init_train',
            init_cfg=init_cfg)

        return r

    def do_cre_nccl(self, cfg={}, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_cre_nccl')
        r = self.do_rpc_call_wkrs(
            'fn_cre_nccl',
            cfg=cfg)

        return r

    def do_cre_nccl_nuid(self, wkr_rank=0, cfg={}, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_cre_nccl_nuid')
        wkr = self.worker_url(wkr_rank)
        cli = self.worker_clis[wkr]
        r = self.do_rpc_call(
            cli, 'fn_cre_nccl_nuid',
            cfg=cfg)

        return r

    def do_init_nccl_comm(self, nuid, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_init_nccl_comm')
        r = self.do_rpc_call_wkrs(
            'fn_init_nccl_comm',
            nuid=nuid)

        return r

    # ============================================================================== from: tndsms
    def do_cre_nccl_nuid_pg(self, wkr_rank=0, cfg={}, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_cre_nccl_nuid_pg')
        wkr = self.worker_url(wkr_rank)
        cli = self.worker_clis[wkr]
        r = self.do_rpc_call(
            cli, 'fn_cre_nccl_nuid_pg',
            cfg=cfg)

        return r

    def do_init_nccl_comm_pg(self, wkr_rank, nuid, pg_rank, world_size, pg_key, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_init_nccl_comm_pg')

        wkr = self.worker_url(wkr_rank)
        cli = self.worker_clis[wkr]
        cfg = {
            'wkr_rank': wkr_rank,
            'nuid': nuid,
            'pg_rank': pg_rank,
            'world_size': world_size,
            'pg_key': pg_key,
        }
        r = self.do_rpc_call(
            cli, 'fn_init_nccl_comm_pg',
            cfg=cfg)

        return r

    def do_check_nccl_groups_cfg_map(self, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_check_nccl_groups_cfg_map')
        r = None  # NOTE: maybe no grp in list
        for pg_grps in self.nccl_groups_grp_list:
            pg_key, grps = pg_grps

            for grp in grps:
                print('nccl_pg_grp', pg_key, grp)
                pg_rank_0 = grp[0]
                pg_cfg = {}

                r = self.do_cre_nccl_nuid_pg(pg_rank_0, pg_cfg)
                ret = r.get('ret')
                msg = r.get('msg', {})
                nuid = msg.get('nuid')

                for i, rnk in enumerate(grp):
                    r = self.do_init_nccl_comm_pg(rnk, nuid, i, len(grp), pg_key)

                time.sleep(1)

            time.sleep(2)

        return r

    # ----------------- init_nccl -----------------





    # ============================================================================== from: tndsms
    def do_one_step_start(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_step_start')
        r = self.do_rpc_call_wkrs(
            'fn_one_step_start',
            cmd=cmd, **kwargs)

        return r

    def fn_one_step_end(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_step_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(11, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_step_end', 'ok')
        return self.msg_pck(rep)


    def do_nccl_all_reduce_start(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_nccl_all_reduce_start')
        r = self.do_rpc_call_wkrs(
            'fn_nccl_all_reduce_start',
            cmd=cmd, **kwargs)

        return r

    def fn_nccl_all_reduce_end(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_nccl_all_reduce_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(12, msg, rank, role)

        rep = self.mk_rpc_ret('fn_nccl_all_reduce_end', 'ok')
        return self.msg_pck(rep)
    # ============================================================================== from: tndsms





    def do_next_epoch_start(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_next_epoch_start')
        r = self.do_rpc_call_wkrs(
            'fn_next_epoch_start',
            cmd=cmd, **kwargs)

        return r

    def fn_next_epoch_end(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_next_epoch_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(13, msg, rank, role)

        rep = self.mk_rpc_ret('fn_next_epoch_end', 'ok')
        return self.msg_pck(rep)

    # ----------------- epoch_func -----------------

    def do_one_mp_step_start(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_start')
        r = self.do_rpc_call_wkrs(
            'fn_one_mp_step_start',
            cmd=cmd, **kwargs)

        return r

    def fn_one_mp_step_ack(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_ack')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 202, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_ack', 'ok')
        return self.msg_pck(rep)

    def fn_one_mp_step_end(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 210, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_end', 'ok')
        return self.msg_pck(rep)



    def _one_mp_step_get_worker_rank_from_net_map(self, step_order_i=None):
        if step_order_i is None:
            step_order_i = self.one_mp_step_order_i
        mod = self.net_map[step_order_i]
        worker_rank = mod.get('worker_rank')
        if worker_rank is None:
            raise Exception('no worker_rank in net_map')
        return worker_rank

    def _one_mp_step_get_worker_ranks_from_grp_map(self, step_order_i=None):
        if step_order_i is None:
            step_order_i = self.one_mp_step_order_i
        ilog.debug(ILOG_I_MP_SVR_SCH_PRT, 'step_order_i', step_order_i)
        worker_ranks = []
        for i in range(self.pp_group_n):
            worker_ranks.append(self.pp_group[i][step_order_i])
        ilog.debug(ILOG_I_MP_SVR_SCH_PRT, 'worker_ranks', worker_ranks)
        return worker_ranks

    def _is_first_part(self, step_order_i=None):
        if step_order_i is None:
            step_order_i = self.one_mp_step_order_i
        return step_order_i == 0

    def _is_last_part(self, step_order_i=None):
        if step_order_i is None:
            step_order_i = self.one_mp_step_order_i
        #return step_order_i == self.one_mp_step_order_n - 1
        return step_order_i == self.pp_group_l - 1

    def do_one_mp_step_forward_________(self, cmd, *args, **kwargs):  # not used now
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_forward')
        wkr_rank = self._one_mp_step_get_worker_rank_from_net_map()
        r = self.do_rpc_call_wkr(
            wkr_rank,
            'fn_one_mp_step_forward',
            cmd=cmd, **kwargs)

        return r

    def do_one_mp_step_forward(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_forward')
        wkr_ranks = self._one_mp_step_get_worker_ranks_from_grp_map()
        r = self.do_rpc_call_some_wkrs(
            wkr_ranks,
            'fn_one_mp_step_forward',
            cmd=cmd, **kwargs)

        return r

    def fn_one_mp_step_forwarded(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_forwarded')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 204, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_forwarded', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_trans_outputs_________(self, cmd, pre_ret, *args, **kwargs):  # not used now
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_trans_outputs')

        output_shps = pre_ret.get('output_shps')
        # TODO: if not output_shps:
        pre_net_map_order = pre_ret.get('net_map_order')
        assert pre_net_map_order == self.one_mp_step_order_i, 'wrong pre_net_map_order'
        from_rank = self._one_mp_step_get_worker_rank_from_net_map(pre_net_map_order)
        to_rank = self._one_mp_step_get_worker_rank_from_net_map(pre_net_map_order + 1)

        r = self.do_rpc_call_wkrs_p2p(
            from_rank, to_rank,
            'fn_one_mp_step_trans_outputs',
            cmd=cmd, output_shps=output_shps, **kwargs)

        return r

    def do_one_mp_step_trans_outputs(self, cmd, pre_ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_trans_outputs')

        output_shps = pre_ret.get('output_shps')
        # TODO: if not output_shps:
        #pre_net_map_order = pre_ret.get('net_map_order')
        #assert pre_net_map_order == self.one_mp_step_order_i, 'wrong pre_net_map_order'
        pre_net_map_order = self.one_mp_step_order_i
        from_ranks = self._one_mp_step_get_worker_ranks_from_grp_map(pre_net_map_order)
        to_ranks = self._one_mp_step_get_worker_ranks_from_grp_map(pre_net_map_order + 1)

        r = self.do_rpc_call_some_wkrs_p2p(
            from_ranks, to_ranks,
            'fn_one_mp_step_trans_outputs',
            cmd=cmd, output_shps=output_shps, **kwargs)

        return r

    def fn_one_mp_step_outputs_transed(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_outputs_transed')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 206, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_outputs_transed', 'ok')
        return self.msg_pck(rep)

    def _save_mp_target_shps(self, ret):
        if self._is_first_part():
            mp_target_shps = ret.get('target_shps')
            self.mp_target_shps = mp_target_shps

    def do_one_mp_step_calc_loss(self, cmd, pre_ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_calc_loss')

        '''
        pre_net_map_order = pre_ret.get('net_map_order')
        assert pre_net_map_order == self.one_mp_step_order_i, 'wrong pre_net_map_order'
        assert self._is_last_part(), 'not is the last part'
        r = self.do_rpc_call_wkr(
            wkr_rank,
            'fn_one_mp_step_calc_loss',
            cmd=cmd, **kwargs)
        '''

        target_shps = self.mp_target_shps
        # TODO: if not target_shps:
        # ==========================================================================
        # NOTE: pre_net_map_order may be the wrong (the first part)
        # ==========================================================================
        #pre_net_map_order = pre_ret.get('net_map_order')
        #assert pre_net_map_order == self.one_mp_step_order_i, 'wrong pre_net_map_order'
        pre_net_map_order = self.one_mp_step_order_i
        assert self._is_last_part(), 'not is the last part'

        '''
        from_rank = self._one_mp_step_get_worker_rank_from_net_map(0)
        to_rank = self._one_mp_step_get_worker_rank_from_net_map(pre_net_map_order)
        r = self.do_rpc_call_wkrs_p2p(
            from_rank, to_rank,
            'fn_one_mp_step_calc_loss',
            cmd=cmd, target_shps=target_shps, **kwargs)
        '''

        from_ranks = self._one_mp_step_get_worker_ranks_from_grp_map(0)
        to_ranks = self._one_mp_step_get_worker_ranks_from_grp_map(pre_net_map_order)
        r = self.do_rpc_call_some_wkrs_p2p(
            from_ranks, to_ranks,
            'fn_one_mp_step_calc_loss',
            cmd=cmd, target_shps=target_shps, **kwargs)

        return r

    def fn_one_mp_step_loss_calced(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_loss_calced')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 222, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_loss_calced', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_trans_grads______________(self, cmd, pre_ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_trans_grads')

        grads_shps = pre_ret.get('grads_shps')
        # TODO: if not grads_shps:
        pre_net_map_order = pre_ret.get('net_map_order')
        assert pre_net_map_order == self.one_mp_step_order_i, 'wrong pre_net_map_order'
        from_rank = self._one_mp_step_get_worker_rank_from_net_map(pre_net_map_order)
        to_rank = self._one_mp_step_get_worker_rank_from_net_map(pre_net_map_order - 1)

        r = self.do_rpc_call_wkrs_p2p(
            from_rank, to_rank,
            'fn_one_mp_step_trans_grads',
            cmd=cmd, grads_shps=grads_shps, **kwargs)

        return r

    def do_one_mp_step_trans_grads(self, cmd, pre_ret, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_trans_grads')

        grads_shps = pre_ret.get('grads_shps')
        # TODO: if not grads_shps:
        #pre_net_map_order = pre_ret.get('net_map_order')
        #assert pre_net_map_order == self.one_mp_step_order_i, 'wrong pre_net_map_order'
        pre_net_map_order = self.one_mp_step_order_i
        from_ranks = self._one_mp_step_get_worker_ranks_from_grp_map(pre_net_map_order)
        to_ranks = self._one_mp_step_get_worker_ranks_from_grp_map(pre_net_map_order - 1)

        r = self.do_rpc_call_some_wkrs_p2p(
            from_ranks, to_ranks,
            'fn_one_mp_step_trans_grads',
            cmd=cmd, grads_shps=grads_shps, **kwargs)

        return r

    def fn_one_mp_step_grads_transed(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_grads_transed')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 224, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_grads_transed', 'ok')
        return self.msg_pck(rep)

    def do_one_mp_step_backward___________(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_backward')
        wkr_rank = self._one_mp_step_get_worker_rank_from_net_map()
        r = self.do_rpc_call_wkr(
            wkr_rank,
            'fn_one_mp_step_backward',
            cmd=cmd, **kwargs)

        return r

    def do_one_mp_step_backward(self, cmd, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_one_mp_step_backward')
        wkr_ranks = self._one_mp_step_get_worker_ranks_from_grp_map()
        r = self.do_rpc_call_some_wkrs(
            wkr_ranks,
            'fn_one_mp_step_backward',
            cmd=cmd, **kwargs)

        return r

    def fn_one_mp_step_backwarded(self, msg, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> fn_one_mp_step_backwarded')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(Q_STATE_HEAD_MP_SCH + 226, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_mp_step_backwarded', 'ok')
        return self.msg_pck(rep)

    def trainer_check_ret(self, ret, rank, role, *args, **kwargs):

        k = str(role) + str(rank)

        r = ret.get('ret')

        if r == 'weights_synced':
            self.weights_sync_flags[k] = 1
            if len(self.weights_sync_flags.keys()) >= self.kn:
                self.weights_sync_flags = {}
                #return 6, 'one_step'
                #return 32, 'dt_sampler_sync'
                return 51, 'distdt_indices'
            else:
                return 0, 'wait'

        elif r == 'dt_sampler_synced':
            self.dt_sampler_sync_flags[k] = 1
            if len(self.dt_sampler_sync_flags.keys()) >= self.kn:
                self.dt_sampler_sync_flags = {}
                return 6, 'one_step'
            else:
                return 0, 'wait'

        elif r == 'one_step':
            self.one_step_flags[k] = 1
            if len(self.one_step_flags.keys()) >= self.kn:
                self.one_step_flags = {}
                return 7, 'nccl_all_reduce'
            else:
                return 0, 'wait'

        # ============================================================================== chg: for dp+mp
        elif r == 'nccl_all_reduced______':  # NOTE: moved to after one_mp_step_backwarded
            self.nccl_all_reduce_flags[k] = 1
            if len(self.nccl_all_reduce_flags.keys()) >= self.kn:
                self.nccl_all_reduce_flags = {}
                return 6, 'one_step'
            else:
                return 0, 'wait'

        # -------------------------------------------------------

        elif r == 'epoch_end':
            self.epoch_end_flags[k] = 1
            if len(self.epoch_end_flags.keys()) >= self.kn:
                self.epoch_end_flags = {}
                self.epoch += 1
                return 8, 'next_epoch'
            else:
                return 0, 'wait'

        elif r == 'next_epoch':
            self.next_epoch_flags[k] = 1
            if len(self.next_epoch_flags.keys()) >= self.kn:
                self.next_epoch_flags = {}
                #return 6, 'one_step'
                #return 32, 'dt_sampler_sync'
                ##return 51, 'distdt_indices'

                self.reset_mp()
                return Q_STATE_HEAD_MP_SCH + 201, 'one_mp_step_start'
            else:
                return 0, 'wait'

        elif r == 'mp_epoch_end________':
            self.epoch += 1
            return 8, 'next_epoch'

        # =======================================================================
        # NOTE: the mp_epoch_end msg will come back from pp_group_n number of
        #       workers (the first parts of MP)
        # =======================================================================
        elif r == 'mp_epoch_end':
            self.mp_epoch_end_flags[k] = 1
            if len(self.mp_epoch_end_flags.keys()) >= self.pp_group_n:
                self.mp_epoch_end_flags = {}
                self.epoch += 1
                return 8, 'next_epoch'
            else:
                return 0, 'wait'

        # -------------------------------------------------------

        elif r == 'one_mp_step_started':
            self.one_mp_step_flags[k] = 1
            if len(self.one_mp_step_flags.keys()) >= self.kn:
                self.one_mp_step_flags = {}
                return Q_STATE_HEAD_MP_SCH + 203, 'one_mp_step_forward'
            else:
                return 0, 'wait'


        elif r == 'one_mp_step_forwarded':
            self.one_mp_step_forward_flags[k] = 1
            if len(self.one_mp_step_forward_flags.keys()) >= self.pp_group_n:
                self.one_mp_step_forward_flags = {}
                if self._is_last_part():
                    # NOTE: ---> loss --> backward
                    return Q_STATE_HEAD_MP_SCH + 221, 'one_mp_step_calc_loss'
                else:
                    self._save_mp_target_shps(ret)
                    return Q_STATE_HEAD_MP_SCH + 205, 'one_mp_step_trans_outputs'
            else:
                return 0, 'wait'

        elif r == 'one_mp_step_outputs_transed':
            self.one_mp_step_trans_outputs_flags[k] = 1
            if len(self.one_mp_step_trans_outputs_flags.keys()) >= (self.pp_group_n * 2):
                self.one_mp_step_trans_outputs_flags = {}
                self.one_mp_step_order_i += 1  # next MP part =====================>
                return Q_STATE_HEAD_MP_SCH + 203, 'one_mp_step_forward'
            else:
                return 0, 'wait'

        # ==========================================================================
        # NOTE: 1. here after calc loss, we do the backward !
        #       2. we use p2p to transmit targets data from first part to last, so
        #          the acks are pp_group_n x2, and we only use ret of last part now
        # ==========================================================================
        elif r == 'one_mp_step_loss_calced________':
            net_map_order = ret.get('net_map_order')
            if self._is_last_part(net_map_order):
                self.one_mp_step_calc_loss_flags[k] = 1
                if len(self.one_mp_step_calc_loss_flags.keys()) >= self.pp_group_n:
                    self.one_mp_step_calc_loss_flags = {}
                    return Q_STATE_HEAD_MP_SCH + 223, 'one_mp_step_trans_grads'
                else:
                    return 0, 'wait'
            else:
                return 0, 'wait'

        elif r == 'one_mp_step_loss_calced':
            self.one_mp_step_calc_loss_flags[k] = 1
            if len(self.one_mp_step_calc_loss_flags.keys()) >= (self.pp_group_n * 2):
                self.one_mp_step_calc_loss_flags = {}
                return Q_STATE_HEAD_MP_SCH + 223, 'one_mp_step_trans_grads'
            else:
                return 0, 'wait'


        elif r == 'one_mp_step_grads_transed':
            self.one_mp_step_trans_grads_flags[k] = 1
            if len(self.one_mp_step_trans_grads_flags.keys()) >= (self.pp_group_n * 2):
                self.one_mp_step_trans_grads_flags = {}
                self.one_mp_step_order_i -= 1  # prev MP part <=====================
                return Q_STATE_HEAD_MP_SCH + 225, 'one_mp_step_backward'
            else:
                return 0, 'wait'

        # ============================================================================== chg: for dp+mp
        elif r == 'one_mp_step_backwarded':
            self.one_mp_step_backward_flags[k] = 1
            if len(self.one_mp_step_backward_flags.keys()) >= self.pp_group_n:
                self.one_mp_step_backward_flags = {}
                if self._is_first_part():
                    #return Q_STATE_HEAD_MP_SCH + 229, 'one_mp_step_end'  # TODO:
                    ##return Q_STATE_HEAD_MP_SCH + 201, 'one_mp_step_start'  # ====>
                    return 7, 'nccl_all_reduce'
                else:
                    return Q_STATE_HEAD_MP_SCH + 223, 'one_mp_step_trans_grads'
            else:
                return 0, 'wait'


        # ============================================================================== chg: for dp+mp
        elif r == 'nccl_all_reduced':  # now for grads all_reduce in DP groups
            self.nccl_all_reduce_flags[k] = 1
            if len(self.nccl_all_reduce_flags.keys()) >= self.kn:
                self.nccl_all_reduce_flags = {}
                #return 6, 'one_step'
                return Q_STATE_HEAD_MP_SCH + 201, 'one_mp_step_start'
            else:
                return 0, 'wait'


        elif r == 'wrong_cmd':
            pass

        else:
            pass

    def check_train_stop(self):
        if self.epoch >= self.epochs:
            return True
        else:
            return False

    def do_stop_train(self, *args, **kwargs):
        ilog.debug(ILOG_I_MP_SVR_SCH, '>>> do_stop_train')
        r = self.do_rpc_call_wkrs(
            'fn_stop_train')

        return r


# ####################################################################################################


class MPSVCScheduler(svctmp.TmplService):
    def __init__(self, cfg={}):
        super(MPSVCScheduler, self).__init__(cfg)

        self.nccl_allreduce_typ = self.cfg.get('nccl_allreduce_typ', 'grad')

        self.init_svr()

        #self.init_train()  # global entry point
        self.wait_for_workers_reg()  # global entry point

    def init_svr(self):
        self.cfg['q'] = self.q
        svr = MPSVRScheduler(self.cfg)
        self.svr = svr

    def init_train(self):
        print('>' * 100)
        # print('>>>', self.q)
        self.q_put_state(1, 'init_train')
        # print('<<<', self.q)

    def wait_for_workers_reg(self):
        self.q_put_state(40, 'wait_wkr_reg')

    # ============================================================================== from: tndsms
    def wait_for_nccl_cre_nuid_pg(self):
        self.q_put_state(900011, 'wait_for_nccl_cre_nuid_pg')

    def check_state(self, *args, **kwargs):

        s = self.q_get_state(timeout=1)
        svr_state, sa, skw = s
        ilog.debug(ILOG_I_MP_SVC_SCH, '=== svr_state', svr_state)

        # NOTE: wait the zrpc call to return
        #self.g_sleep(0.001)

        if svr_state == 0:
            r = self.svr.check_train_stop()
            if r:
                self.q_put_state(20, 'train_stop')


        elif svr_state == 40:
            r = self.svr.check_workers_reg()
            # TODO: if all workers registered
            #self.q_put_state(1, 'init_train')
            self.g_sleep(1.0)

        elif svr_state == 41:
            if sa:
                msg, rank, role = sa
                wkr_url = msg.get('url')

                r = self.svr.check_workers_reg(rank, role, msg)

                if r:  # all workers have already registered
                    self.init_train()
                else:
                    self.wait_for_workers_reg()
            else:
                pass

        # ----------------- worker_reg -----------------

        elif svr_state == 1:
            r = self.svr.do_init_train()
            self.q_put_state(2, 'cre_nccl')

        elif svr_state == 2:
            r = self.svr.do_cre_nccl()
            self.q_put_state(3, 'cre_nccl_nuid')


        elif svr_state == 3:
            r = self.svr.do_cre_nccl_nuid()
            ret = r.get('ret')
            # TODO: check ret
            msg = r.get('msg', {})
            nuid = msg.get('nuid')

            r = self.svr.do_init_nccl_comm(nuid)

            #self.q_put_state(5, 'start_epoch')
            #self.q_put_state(6, 'one_step')
            ###self.q_put_state(30, 'weights_sync')  # TODO: =============>
            ##self.q_put_state(51, 'distdt_indices')
            #self.q_put_state(Q_STATE_HEAD_MP_SCH + 201, 'one_mp_step_start')  # ==========>
            self.q_put_state(900011, 'check_nccl_groups_cfg_map')

        elif svr_state == 900011:
            self.g_sleep(1.0)  # NOTE: waiting for setting main nk in ncclgrp

            r = self.svr.do_check_nccl_groups_cfg_map()

            #self.q_put_state(5, 'start_epoch')
            #self.q_put_state(6, 'one_step')
            #self.q_put_state(30, 'weights_sync')  # ==========>
            ##self.q_put_state(51, 'distdt_indices')
            self.q_put_state(Q_STATE_HEAD_MP_SCH + 201, 'one_mp_step_start')

        # ----------------- init_nccl -----------------


        # ============================================================================== from: tndsms
        elif svr_state == 6:

            # NOTE: wait the zrpc call to return (for dt_sampler_sync)
            #self.g_sleep(0.001)  # to <3.3>

            cmd = sa[0]
            r = self.svr.do_one_step_start(cmd, typ=self.nccl_allreduce_typ)

        elif svr_state == 7:
            cmd = sa[0]
            r = self.svr.do_nccl_all_reduce_start(cmd, typ=self.nccl_allreduce_typ)
        # ============================================================================== from: tndsms



        elif svr_state == 8:
            cmd = sa[0]
            r = self.svr.do_next_epoch_start(cmd)

        # ============================================================================== from: tndsms
        elif svr_state == 11 or svr_state == 12 or svr_state == 13:
        #elif svr_state == 13:
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t)
            else:
                pass

        # ----------------- epoch_func -----------------

        elif svr_state == Q_STATE_HEAD_MP_SCH + 201:  # one_mp_step_start
            cmd = sa[0]
            r = self.svr.do_one_mp_step_start(cmd)

        elif svr_state == Q_STATE_HEAD_MP_SCH + 202:  # fn_one_mp_step_ack
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t)
            else:
                pass

        elif svr_state == Q_STATE_HEAD_MP_SCH + 203:  # one_mp_step_forward
            cmd = sa[0]
            r = self.svr.do_one_mp_step_forward(cmd)

        elif svr_state == Q_STATE_HEAD_MP_SCH + 204:  # fn_one_mp_step_forwarded
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t, ret)
            else:
                pass

        elif svr_state == Q_STATE_HEAD_MP_SCH + 205:  # one_mp_step_trans_outputs
            cmd = sa[0]
            pre_ret = sa[1]
            r = self.svr.do_one_mp_step_trans_outputs(cmd, pre_ret)

        elif svr_state == Q_STATE_HEAD_MP_SCH + 206:  # fn_one_mp_step_outputs_transed
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t, ret)
            else:
                pass

        # >>>>>>>>>>>-------------<<<<<<<<<<<

        elif svr_state == Q_STATE_HEAD_MP_SCH + 221:  # one_mp_step_calc_loss
            cmd = sa[0]
            pre_ret = sa[1]
            r = self.svr.do_one_mp_step_calc_loss(cmd, pre_ret)

        elif svr_state == Q_STATE_HEAD_MP_SCH + 222:  # fn_one_mp_step_loss_calced
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t, ret)
            else:
                pass

        elif svr_state == Q_STATE_HEAD_MP_SCH + 223:  # one_mp_step_trans_grads
            cmd = sa[0]
            pre_ret = sa[1]
            r = self.svr.do_one_mp_step_trans_grads(cmd, pre_ret)

        elif svr_state == Q_STATE_HEAD_MP_SCH + 224:  # fn_one_mp_step_grads_transed
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t, ret)
            else:
                pass

        elif svr_state == Q_STATE_HEAD_MP_SCH + 225:  # one_mp_step_backward
            cmd = sa[0]
            pre_ret = sa[1]
            r = self.svr.do_one_mp_step_backward(cmd, pre_ret)

        elif svr_state == Q_STATE_HEAD_MP_SCH + 226:  # fn_one_mp_step_backwarded
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t, ret)
            else:
                pass

        # --------------------------------------------------------

        elif svr_state == 20:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.01)

            r = self.svr.do_stop_train()

            # NOTE: wait the zrpc call to return
            self.g_sleep(3.0)

            self.stop()  # service loop stop

        else:
            r = self.svr.check_train_stop()
            if r:
                self.q_put_state(20, 'train_stop')

