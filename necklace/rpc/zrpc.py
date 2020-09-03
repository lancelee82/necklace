"""ZeroRPC Base"""
from __future__ import print_function

import sys
import uuid
import time
from collections import OrderedDict

import multiprocessing as mp
import gevent
'''
# TODO: the multiprocessing and gevent are conflict [?]
from gevent import monkey
#monkey.patch_all()
##monkey.patch_subprocess()
#monkey.patch_all(thread=False, socket=False)
monkey.patch_all(socket=False)
##monkey.patch_socket()
'''

from zerorpc import zmq
import zerorpc

from . import msgpck
from . import ilog

import logging
logging.getLogger("zerorpc").setLevel(logging.WARNING)


class NodeBase(object):
    pass


class NodeMixin(NodeBase):
    def __init__(self, ndid=None, cfg={}):
        self.ndid = ndid
        if not self.ndid:
            self.ndid = cfg.get('ndid')
        if not self.ndid:
            self.ndid = str(uuid.uuid4())

        self.cfg = cfg


class Server(NodeMixin, zerorpc.Server):
    def __init__(self, ndid=None, cfg={}):
        super(Server, self).__init__(ndid, cfg)

        # \zerorpc-python\zerorpc\core.py
        #def __init__(self, methods=None, name=None,
        #             context=None, pool_size=None,
        #             heartbeat=5):
        zerorpc.Server.__init__(self)

        self.stopped = False

        self.q = self.cfg.get('q')
        if self.q is None:
            self.q = gevent.queue.Queue()

        self.init_sock()

    def init_sock(self):
        self.url = self.cfg.get('url') or 'ipc:///tmp/zrpc-1.ipc'

        self.bind(self.url)

    def q_put(self, q=None, m=None, *args, **kwargs):
        if q is None:
            q = self.q

        q.put(m)

    def q_get(self, q=None, *args, **kwargs):
        if q is None:
            q = self.q

        try:
            r = q.get(*args, **kwargs)
        except Exception as e:
            r = None
        return r

    def msg_pck(self, msg):
        return msgpck.packb(msg)

    def msg_unpck(self, msg):
        return msgpck.unpackb(msg)

    def msg_pkpck(self, msg):
        return msgpck.pkpackb(msg)

    def msg_unpkpck(self, msg):
        return msgpck.unpkpackb(msg)

    def snd_msg(self, cli, func, *args, **kwargs):
        ret = getattr(cli, func)(*args, **kwargs)
        return ret

    def loop(self):
        self.run()

    # api for example
    def lolita(self):
        return 42

    def add(self, a, b):
        return a + b


class Client(NodeMixin, zerorpc.Client):
    def __init__(self, ndid=None, cfg={}):
        super(Client, self).__init__(ndid, cfg)

        # \zerorpc-python\zerorpc\core.py
        #def __init__(self, connect_to=None, context=None,
        #             timeout=30, heartbeat=5,
        #             passive_heartbeat=False):
        timeout = self.cfg.get('zc_hb_timeout', 100)
        heartbeat = self.cfg.get('zc_hb_heartbeat', 10)
        zerorpc.Client.__init__(self, timeout=timeout, heartbeat=heartbeat)

        self.init_sock()

    def init_sock(self):
        self.url = self.cfg.get('url') or 'ipc:///tmp/zrpc-1.ipc'

        self.connect(self.url)

    def msg_pck(self, msg):
        return msgpck.packb(msg)

    def msg_unpck(self, msg):
        return msgpck.unpackb(msg)

    def msg_pkpck(self, msg):
        return msgpck.pkpackb(msg)

    def msg_unpkpck(self, msg):
        return msgpck.unpkpackb(msg)

    def snd_msg(self, func, *args, **kwargs):
        ret = getattr(self, func)(*args, **kwargs)
        return ret

    # to be override
    def loop(self):
        pass


class Node(object):
    r"""A complete functional RPC Node.

    A Node contains two main parts:
    1. a server which is binding a port and listening connections from the clients;
    2. some timers those execute jobs from the clients.
    The connection between the server and the timers is a queue, and this job queue
    make the two parts could do jobs separately like multi threads, but note that
    the server is really in a greenlet.
    """

    def __init__(self, cfg, svr=None):
        self.cfg = cfg

        self.svr = svr  # an instance of the class Server

        self.clis = {}  # some instances of the class Client

        self.timers = OrderedDict()  # {}

        self.state = 0
        self.stopped = False

        self.q = gevent.queue.Queue()

        self.init_clis()

        self.init_timers()

    def init_clis(self):
        clis_e = self.cfg.get('clis')
        if clis_e:
            for cl in clis_e:
                cfg = {
                    'url': cl,
                }
                c = Client(cl, cfg)
                # NOTE: here we use the url of the client as the key
                self.clis[cl] = c

    # overwrite
    def init_timers(self):
        # use register_timer
        self.register_timer(0, self.check_state)

    def func_name(self, func):
        fname = str(id(func)) + str(uuid.uuid4())
        return fname

    def register_timer(self, intvl, func, *args, **kwargs):
        fname = self.func_name(func)
        tm = time.time()
        self.timers[fname] = [tm, intvl, func, args, kwargs]

    # overwrite
    def check_state(self, *args, **kwargs):
        pass  # do nothing for now

    # NOTE: the first item of m should be the state of svr,
    # then it could be checked in check_state (implemented in svctmp.py now)
    def q_put(self, q=None, m=None, *args, **kwargs):
        if q is None:
            q = self.q

        q.put(m)

    def q_get(self, q=None, *args, **kwargs):
        if q is None:
            q = self.q

        try:
            r = q.get(*args, **kwargs)
        except Exception as e:
            #print(e)
            r = None
        return r

    def g_sleep(self, n):
        gevent.sleep(n)

    def tm_to_sleep(self):
        if not self.timers:
            return 1
        else:
            # return 0
            # or,
            # get the min time from timers
            tms = [v[1] for k, v in self.timers.items()]
            tms.sort()
            tm = tms[0] / 2.0
            return tm

    def run_timers(self):
        now = time.time()
        for k, v in self.timers.items():

            # NOTE: make the zerorpc could do other things
            self.g_sleep(0)

            tm = v[0]
            intvl = v[1]
            if now - tm > intvl:
                func = v[2]
                args = v[3]
                kwargs = v[4]
                func(*args, **kwargs)
                v[0] = now

                self.g_sleep(0)

    def stop(self):
        self.stopped = True

    def start_svr(self):
        if self.svr:
            gevent.spawn(self.svr.run)

    def loop(self):
        # <1.1> create a server (binding a port) in a greenlet and do listening,
        #       as when the server accepts one connection, it will
        #       (1. rpc-cast) unpack the msg and put it into the queue
        #       which be waiting for the service to check it in one of the timers,
        #       then return an ack to client immediately;
        #       (2. rpc-call) do the work at local and then return
        #       the result to client, note: the rpc-call will block the server
        #       and so it is not satisfied for long-run work
        self.start_svr()

        while not self.stopped:
            # <1.2> periodically execute the timers in which check the msg from
            #       the queue and do the works those have been defined
            self.run_timers()
            gevent.sleep(self.tm_to_sleep())
