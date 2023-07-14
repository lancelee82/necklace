"""Distributed Trainer Services using Numba and NCCL (gradients allreduce)"""
from __future__ import print_function, absolute_import, division

import time
import uuid
from collections import OrderedDict

from necklace.rpc import zrpc
from necklace.rpc import svctmp
from necklace.rpc import msgpck
from necklace.rpc import ilog


class SVRWorker(svctmp.TmplServer):
    def __init__(self, cfg={}):
        super(SVRWorker, self).__init__(cfg)

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
        ilog.debug(500001, '>>> do_worker_reg')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_worker_reg',
            url=self.url)
        return r

    def fn_init_train(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_init_train')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(1, msg)

        rep = self.mk_rpc_ret('fn_init_train', 'ok')
        return self.msg_pck(rep)

    def fn_cre_nccl(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_cre_nccl')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(2, msg)

        rep = self.mk_rpc_ret('fn_cre_nccl', 'ok')
        return self.msg_pck(rep)

    def fn_cre_nccl_nuid(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_cre_nccl_nuid')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        #self.q_put_state(3, msg)
        nuid = self.trnr.cre_nccl_nuid(msg)
        nuid = msgpck.pkl_dumps(nuid)

        rep = self.mk_rpc_ret('fn_cre_nccl_nuid', 'ok', nuid=nuid)
        return self.msg_pck(rep)

    def fn_init_nccl_comm(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_init_nccl_comm')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(4, msg)

        rep = self.mk_rpc_ret('fn_init_nccl_comm', 'ok')
        return self.msg_pck(rep)

    def fn_cre_distdt_indices(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_cre_distdt_indices')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        dt_indices = self.trnr.cre_distdt_indices(msg)
        #dt_indices = msgpck.pkl_dumps(dt_indices)

        rep = self.mk_rpc_ret('fn_cre_distdt_indices', 'ok', dt_indices=dt_indices)
        return self.msg_pck(rep)

    def fn_reset_distdt_indices(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_reset_distdt_indices')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(52, msg)

        rep = self.mk_rpc_ret('fn_reset_distdt_indices', 'ok')
        return self.msg_pck(rep)


    def fn_weights_sync_start(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_weights_sync_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(31, msg)

        rep = self.mk_rpc_ret('fn_weights_sync_start', 'ok')
        return self.msg_pck(rep)

    def do_weights_sync_end(self, ret, *args, **kwargs):
        ilog.debug(500001, '>>> do_weights_sync_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_weights_sync_end',
            ret=ret)
        return r


    def fn_dt_sampler_sync_start(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_dt_sampler_sync_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(33, msg)

        rep = self.mk_rpc_ret('fn_dt_sampler_sync_start', 'ok')
        return self.msg_pck(rep)

    def do_dt_sampler_sync_end(self, ret, *args, **kwargs):
        ilog.debug(500001, '>>> do_dt_sampler_sync_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_dt_sampler_sync_end',
            ret=ret)
        return r


    def fn_start_epoch(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_start_epoch')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(5, msg)
        #nuid = self.trnr.start_epoch()

        rep = self.mk_rpc_ret('fn_start_epoch', 'ok')
        return self.msg_pck(rep)


    def fn_one_step(self, msg, *args, **kwargs):  # -x-
        ilog.debug(500001, '>>> fn_one_step')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        #self.q_put_state(6, msg)
        #r = self.trnr.one_step()
        r = self.trnr.check_msg(msg)

        rep = self.mk_rpc_ret('fn_one_step', 'ok', r=r)
        return self.msg_pck(rep)

    def fn_nccl_all_reduce(self, msg, *args, **kwargs):  # -x-
        ilog.debug(500001, '>>> fn_nccl_all_reduce')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        #self.q_put_state(7, msg)
        #r = self.trnr.one_step()
        r = self.trnr.check_msg(msg)

        rep = self.mk_rpc_ret('fn_nccl_all_reduce', 'ok', r=r)
        return self.msg_pck(rep)

    def fn_next_epoch(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_next_epoch')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        #self.q_put_state(8, msg)
        #r = self.trnr.one_step()
        r = self.trnr.check_msg(msg)

        rep = self.mk_rpc_ret('fn_next_epoch', 'ok', r=r)
        return self.msg_pck(rep)


    def fn_one_step_start(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_one_step_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(11, msg)
        #r = self.trnr.one_step()
        #r = self.trnr.check_msg(msg)

        rep = self.mk_rpc_ret('fn_one_step_start', 'ok')
        return self.msg_pck(rep)

    def fn_nccl_all_reduce_start(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_nccl_all_reduce_start')
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
        ilog.debug(500001, '>>> do_one_step_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_one_step_end',
            ret=ret)
        return r

    def do_nccl_all_reduce_end(self, ret, *args, **kwargs):
        ilog.debug(500001, '>>> do_nccl_all_reduce_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_nccl_all_reduce_end',
            ret=ret)
        return r


    def fn_next_epoch_start(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_next_epoch_start')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(13, msg)

        rep = self.mk_rpc_ret('fn_next_epoch_start', 'ok')
        return self.msg_pck(rep)

    def do_next_epoch_end(self, ret, *args, **kwargs):
        ilog.debug(500001, '>>> do_next_epoch_end')
        r = self.do_rpc_call(
            self.scheduler_cli, 'fn_next_epoch_end',
            ret=ret)
        return r


    def fn_stop_train(self, msg, *args, **kwargs):
        ilog.debug(500001, '>>> fn_stop_train')
        m = self.msg_unpck(msg)
        msg = m.get('msg') or {}

        self.q_put_state(20, msg)
        #self.trainer_stop()

        rep = self.mk_rpc_ret('fn_stop_train', 'ok')
        return self.msg_pck(rep)

    def trainer_stop(self):
        r = self.trnr.clear()


# ============================================================================


class SVCWorker(svctmp.TmplService):
    def __init__(self, cfg={}):
        super(SVCWorker, self).__init__(cfg)

        self.init_svr()

        self.reg_worker()  # global entry

    def init_svr(self):
        self.cfg['q'] = self.q
        svr = SVRWorker(self.cfg)
        self.svr = svr

    def reg_worker(self):
        self.q_put_state(40, 'worker_reg')

    def check_state(self, *args, **kwargs):

        s = self.q_get_state(timeout=1)
        svr_state, sa, skw = s
        ilog.debug(500000, '=== svr_state', self.svr.rank, svr_state)

        if svr_state == 0:
            pass


        elif svr_state == 40:  # worker_reg
            self.svr.do_worker_reg()


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


        elif svr_state == 52:
            msg = sa[0]
            dt_indices = msg.get('dt_indices')
            #dt_indices = msgpck.pkl_loads(dt_indices)
            msg['dt_indices'] = dt_indices

            # NOTE: wait the zrpc call to return
            self.g_sleep(1.0)

            r = self.svr.trnr.reset_distdt_indices(msg)
            #self.q_put_state(0, r)


        elif svr_state == 5:  # start_epoch
            pass


        elif svr_state == 31:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.0001)

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_weights_sync_end(ret)
            #self.q_put_state(0, r)

        elif svr_state == 33:

            # NOTE: wait the zrpc call to return
            self.g_sleep(0.0001)

            msg = sa[0]
            ret = self.svr.trnr.check_msg(msg)
            r = self.svr.do_dt_sampler_sync_end(ret)
            #self.q_put_state(0, r)


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

        else:
            pass


# ============================================================================
# ============================================================================
# ============================================================================


class SVRScheduler(svctmp.TmplServer):
    def __init__(self, cfg={}):
        super(SVRScheduler, self).__init__(cfg)

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

        #self.worker_urls = self.cfg.get('worker_urls')
        #self.init_worker_clis(self.worker_urls)
        # --> wait for worker to register themself

        #self.wdsz = self.cfg.get('wdsz')

        self.workers = self.cfg.get('workers')
        self.worker_eps = OrderedDict()

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
        ilog.debug(500002, '>>> fn_worker_reg')
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

    def do_init_train(self, init_cfg={}, *args, **kwargs):
        ilog.debug(500002, '>>> do_init_train')
        r = self.do_rpc_call_wkrs(
            'fn_init_train',
            init_cfg=init_cfg)

        return r

    def do_cre_nccl(self, cfg={}, *args, **kwargs):
        ilog.debug(500002, '>>> do_cre_nccl')
        r = self.do_rpc_call_wkrs(
            'fn_cre_nccl',
            cfg=cfg)

        return r

    def do_cre_nccl_nuid(self, wkr_rank=0, cfg={}, *args, **kwargs):
        ilog.debug(500002, '>>> do_cre_nccl_nuid')
        wkr = self.worker_url(wkr_rank)
        cli = self.worker_clis[wkr]
        r = self.do_rpc_call(
            cli, 'fn_cre_nccl_nuid',
            cfg=cfg)

        return r

    def do_init_nccl_comm(self, nuid, *args, **kwargs):
        ilog.debug(500002, '>>> do_init_nccl_comm')
        r = self.do_rpc_call_wkrs(
            'fn_init_nccl_comm',
            nuid=nuid)

        return r

    def do_cre_distdt_indices(self, wkr_rank=0, cfg={}, *args, **kwargs):
        ilog.debug(500002, '>>> do_cre_distdt_indices')
        wkr = self.worker_url(wkr_rank)
        cli = self.worker_clis[wkr]
        r = self.do_rpc_call(
            cli, 'fn_cre_distdt_indices',
            cfg=cfg)

        return r

    def do_reset_distdt_indices(self, dt_indices, *args, **kwargs):
        ilog.debug(500002, '>>> do_reset_distdt_indices')
        r = self.do_rpc_call_wkrs(
            'fn_reset_distdt_indices',
            dt_indices=dt_indices)

        return r

    def do_weights_sync_start(self, cmd, *args, **kwargs):
        ilog.debug(500002, '>>> do_weights_sync_start')
        r = self.do_rpc_call_wkrs(
            'fn_weights_sync_start',
            cmd=cmd, **kwargs)

        return r

    def fn_weights_sync_end(self, msg, *args, **kwargs):
        ilog.debug(500002, '>>> fn_weights_sync_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(31, msg, rank, role)

        rep = self.mk_rpc_ret('fn_weights_sync_end', 'ok')
        return self.msg_pck(rep)


    def do_dt_sampler_sync_start(self, cmd, *args, **kwargs):
        ilog.debug(500002, '>>> do_dt_sampler_sync_start')
        r = self.do_rpc_call_wkrs(
            'fn_dt_sampler_sync_start',
            cmd=cmd, **kwargs)

        return r

    def fn_dt_sampler_sync_end(self, msg, *args, **kwargs):
        ilog.debug(500002, '>>> fn_dt_sampler_sync_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(33, msg, rank, role)

        rep = self.mk_rpc_ret('fn_dt_sampler_sync_end', 'ok')
        return self.msg_pck(rep)


    def do_one_step_start(self, cmd, *args, **kwargs):
        ilog.debug(500002, '>>> do_one_step_start')
        r = self.do_rpc_call_wkrs(
            'fn_one_step_start',
            cmd=cmd, **kwargs)

        return r

    def fn_one_step_end(self, msg, *args, **kwargs):
        ilog.debug(500002, '>>> fn_one_step_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(11, msg, rank, role)

        rep = self.mk_rpc_ret('fn_one_step_end', 'ok')
        return self.msg_pck(rep)


    def do_nccl_all_reduce_start(self, cmd, *args, **kwargs):
        ilog.debug(500002, '>>> do_nccl_all_reduce_start')
        r = self.do_rpc_call_wkrs(
            'fn_nccl_all_reduce_start',
            cmd=cmd, **kwargs)

        return r

    def fn_nccl_all_reduce_end(self, msg, *args, **kwargs):
        ilog.debug(500002, '>>> fn_nccl_all_reduce_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(12, msg, rank, role)

        rep = self.mk_rpc_ret('fn_nccl_all_reduce_end', 'ok')
        return self.msg_pck(rep)


    def do_next_epoch_start(self, cmd, *args, **kwargs):
        ilog.debug(500002, '>>> do_next_epoch_start')
        r = self.do_rpc_call_wkrs(
            'fn_next_epoch_start',
            cmd=cmd, **kwargs)

        return r

    def fn_next_epoch_end(self, msg, *args, **kwargs):
        ilog.debug(500002, '>>> fn_next_epoch_end')
        m = self.msg_unpck(msg)
        role = m.get('role')
        rank = m.get('rank')
        msg = m.get('msg') or {}
        # TODO: do check

        self.q_put_state(13, msg, rank, role)

        rep = self.mk_rpc_ret('fn_next_epoch_end', 'ok')
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

        elif r == 'epoch_end':
            self.epoch_end_flags[k] = 1
            if len(self.epoch_end_flags.keys()) >= self.kn:
                self.epoch_end_flags = {}
                self.epoch += 1
                return 8, 'next_epoch'
            else:
                return 0, 'wait'

        elif r == 'nccl_all_reduced':
            self.nccl_all_reduce_flags[k] = 1
            if len(self.nccl_all_reduce_flags.keys()) >= self.kn:
                self.nccl_all_reduce_flags = {}
                return 6, 'one_step'
            else:
                return 0, 'wait'

        elif r == 'next_epoch':
            self.next_epoch_flags[k] = 1
            if len(self.next_epoch_flags.keys()) >= self.kn:
                self.next_epoch_flags = {}
                #return 6, 'one_step'
                #return 32, 'dt_sampler_sync'
                return 51, 'distdt_indices'
            else:
                return 0, 'wait'

        elif r == 'wrong_cmd':
            pass

        else:
            pass


    def do_next_epoch(self, cmd, *args, **kwargs):  # not used now
        ilog.debug(500002, '>>> do_next_epoch')
        r = self.do_rpc_call_wkrs(
            'fn_next_epoch',
            cmd=cmd, **kwargs)

        return r

    def check_train_stop(self):
        if self.epoch >= self.epochs:
            return True
        else:
            return False

    def do_stop_train(self, *args, **kwargs):
        ilog.debug(500002, '>>> do_stop_train')
        r = self.do_rpc_call_wkrs(
            'fn_stop_train')

        return r


# ============================================================================


class SVCScheduler(svctmp.TmplService):
    def __init__(self, cfg={}):
        super(SVCScheduler, self).__init__(cfg)

        self.nccl_allreduce_typ = self.cfg.get('nccl_allreduce_typ', 'grad')

        self.init_svr()

        #self.init_train()  # global entry point
        self.wait_for_workers_reg()  # global entry point

    def init_svr(self):
        self.cfg['q'] = self.q
        svr = SVRScheduler(self.cfg)
        self.svr = svr

    def init_train(self):
        print('>' * 100)
        # print('>>>', self.q)
        self.q_put_state(1, 'init_train')
        # print('<<<', self.q)

    def wait_for_workers_reg(self):
        self.q_put_state(40, 'wait_wkr_reg')

    def check_state(self, *args, **kwargs):

        s = self.q_get_state(timeout=1)
        svr_state, sa, skw = s
        ilog.debug(500003, '=== svr_state', svr_state)

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
            self.q_put_state(30, 'weights_sync')
            ##self.q_put_state(51, 'distdt_indices')


        elif svr_state == 51:  # do every epoch
            r = self.svr.do_cre_distdt_indices()  # NOTE: only call rank-0
            ret = r.get('ret')
            # TODO: check ret
            msg = r.get('msg', {})
            dt_indices = msg.get('dt_indices')

            r = self.svr.do_reset_distdt_indices(dt_indices)

            #self.q_put_state(5, 'start_epoch')
            self.q_put_state(6, 'one_step')
            ##self.q_put_state(30, 'weights_sync')


        elif svr_state == 4:
            pass
        elif svr_state == 5:
            pass


        elif svr_state == 30:
            cmd = sa[0]
            r = self.svr.do_weights_sync_start(cmd, typ=self.nccl_allreduce_typ)

        elif svr_state == 31:
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t)
            else:
                pass


        elif svr_state == 32:
            cmd = sa[0]
            r = self.svr.do_dt_sampler_sync_start(cmd, typ=self.nccl_allreduce_typ)

        elif svr_state == 33:
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t)

                self.g_sleep(0.001)  # NOTE: wait (for dt_sampler_sync) <3.3>
            else:
                pass


        elif svr_state == 6:

            # NOTE: wait the zrpc call to return (for dt_sampler_sync)
            #self.g_sleep(0.001)  # to <3.3>

            cmd = sa[0]
            r = self.svr.do_one_step_start(cmd, typ=self.nccl_allreduce_typ)

        elif svr_state == 7:
            cmd = sa[0]
            r = self.svr.do_nccl_all_reduce_start(cmd, typ=self.nccl_allreduce_typ)


        elif svr_state == 8:
            cmd = sa[0]
            #r = self.svr.do_next_epoch(cmd, typ=self.nccl_allreduce_typ)
            ##self.q_put_state(5, 'start_epoch')
            ##self.q_put_state(6, 'one_step')  # ==> start a new epoch
            #self.q_put_state(32, 'dt_sampler_sync')  # ==> resync data sampler
            r = self.svr.do_next_epoch_start(cmd, typ=self.nccl_allreduce_typ)


        elif svr_state == 11 or svr_state == 12 or svr_state == 13:
            if sa:
                msg, rank, role = sa
                ret = msg.get('ret')

                s, t = self.svr.trainer_check_ret(ret, rank, role)

                self.q_put_state(s, t)
            else:
                pass

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

