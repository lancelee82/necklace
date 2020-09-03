"""zRPC based Server/Service Template"""
from __future__ import print_function

import time
import uuid
from collections import OrderedDict

#import multiprocessing as mp
from .zrpc import mp

from . import zrpc
from . import msgpck
from . import ilog


class TmplClient(zrpc.Client):
    def __init__(self, cfg={}):
        rank = cfg.get('rank')
        if rank is None:
            rank = 0
        self.rank = rank
        ndid = 'cli-' + str(self.rank)
        super(TmplClient, self).__init__(ndid, cfg)


class TmplServer(zrpc.Server):
    def __init__(self, cfg={}):
        rank = cfg.get('rank')
        role = cfg.get('role')
        if rank is None:
            rank = 0
        self.rank = rank
        self.role = role
        ndid = 'svr-' + str(self.rank)
        super(TmplServer, self).__init__(ndid, cfg)

        self.state = 0

    def q_put_state(self, state, *args, **kwargs):
        self.state = state
        self.q_put(self.q, (self.state, args, kwargs))

    def mk_rpc_cli(self, cli_url):
        cfg = {
            'url': cli_url,
        }
        cli = TmplClient(cfg)
        return cli

    def mk_rpc_msg(self, fn, *args, **kwargs):

        m = {
            'fn': fn,
            'role': self.role,
            'ndid': self.ndid,
            'rank': self.rank,
            'url': self.url,
            'msg': kwargs,
        }

        return m

    def mk_rpc_ret(self, fn, ret, *args, **kwargs):

        m = {
            'fn': fn,
            'role': self.role,
            'ndid': self.ndid,
            'rank': self.rank,
            'url': self.url,
            'ret': ret,
            'msg': kwargs,
        }

        return m

    def rep_err_msg(self, fn, msg):
        ret = {
            'fn': fn,
            'ret': 'err',
            'msg': msg,
        }
        return self.msg_pck(ret)

    def do_rpc_call(self, cli, fn, *args, **kwargs):

        m = {
            'fn': fn,
            'role': self.role,
            'ndid': self.ndid,
            'rank': self.rank,
            'url': self.url,
            'msg': kwargs,
        }

        msg = self.msg_pck(m)
        ret = self.snd_msg(cli, fn, msg)  # --> rpc call
        r = self.msg_unpck(ret)

        return r


class TmplService(zrpc.Node):
    def __init__(self, cfg={}):
        super(TmplService, self).__init__(cfg)

    def q_put_state(self, state, *args, **kwargs):
        self.svr.state = state
        self.q_put(self.q, (self.svr.state, args, kwargs))

    def q_get_state(self, *args, **kwargs):
        s = self.q_get(self.q, *args, **kwargs)
        if not s:
            return None, None, None
        else:
            return s  # self.state, args, kwargs

    def init_timers(self):
        # now we only add one timer here and do all things in it
        self.register_timer(0, self.check_state)

    # overwrite
    def check_state(self, *args, **kwargs):
        pass  # overwrite this function and do every things here
