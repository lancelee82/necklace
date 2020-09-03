"""Base Trainer for One Process"""
from __future__ import print_function, absolute_import, division

import copy
import random
import time
from collections import OrderedDict
import uuid
import itertools

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

from necklace.cuda import nbutils
from necklace.cuda import ncclwrp


class TrainerOPAbs(object):
    def __init__(self, cfg={}):
        self.cfg = cfg


    def before_train(self, rank=0, *args, **kwargs):
        pass


    def before_epoch(self, rank=0, *args, **kwargs):
        pass


    def get_weights(self):  # --> weights
        pass


    def set_weights(self, weights):
        pass


    def get_optimizer(self, model=None, name=None):  # --> optimz
        pass


    def init_data_loader_wrapper(self, rank=0, typ='pt', *args, **kwargs):
        pass


    def get_a_data_batch_index(self, rank=0, *args, **kwargs):  # --> [i1, ix, ...]
        pass


    def data_batchify(self, data, *args, **kwargs):
        pass


    def one_step(self, rank=0, *args, **kwargs):
        pass


    def get_a_data_batch(self, rank=0, *args, **kwargs):  # --> x, y
        pass


    def train_a_batch_grads(self, rank=0, *args, **kwargs):  # --> gradients
        pass


    def do_a_batch_grads(self, inputs, targets, loss_fns, *args, **kwargs):
        pass


    def do_a_batch_loss(self, outputs, targets, loss_fns, *args, **kwargs):
        pass


    def collect_gradients(self, *args, **kwargs):  # --> gradients
        pass


    def accumulate_gradients(self, grads_now, grads_new):  # --> gradients
        pass


    def apply_gradients(self, gradients, n_grads):  # --> weights
        pass


    def do_a_train_step(self, stp=None):  # == apply_gradients
        pass


    def after_train(self, rank=0, *args, **kwargs):
        pass


    def after_epoch(self, rank=0, *args, **kwargs):
        pass


    def if_stop(self):
        pass


    def eval_a(self):
        pass


    def start(self, role=None):
        pass


class ModelWrapperHooks(TrainerOPAbs):
    def __init__(self, cfg={}):
        super(ModelWrapperHooks, self).__init__(cfg)

        self.hook_before_train_funcs = OrderedDict()
        self.hook_after_train_funcs = OrderedDict()

        self.hook_before_epoch_funcs = OrderedDict()
        self.hook_after_epoch_funcs = OrderedDict()

        self.hook_before_data_batch_funcs = OrderedDict()
        self.hook_after_data_batch_funcs = OrderedDict()

        self.hook_before_batch_funcs = OrderedDict()
        self.hook_after_batch_funcs = OrderedDict()

        self.hook_before_do_a_batch_grads_funcs = OrderedDict()
        self.hook_after_do_a_batch_grads_funcs = OrderedDict()

        self.hook_before_do_a_train_step_funcs = OrderedDict()
        self.hook_after_do_a_train_step_funcs = OrderedDict()

        self.hook_vars = {}  # for hook funcs use


    def before_train(self, rank=0, *args, **kwargs):
        self.hook_before_train()
        self._before_train(rank, *args, **kwargs)

    def after_train(self, rank=0, *args, **kwargs):
        self._after_train(rank, *args, **kwargs)
        self.hook_after_train()

    def before_epoch(self, rank=0, *args, **kwargs):
        self.hook_before_epoch()
        self._before_epoch(rank, *args, **kwargs)

    def after_epoch(self, rank=0, *args, **kwargs):
        self._after_epoch(rank, *args, **kwargs)
        self.hook_after_epoch()


    def get_a_data_batch(self, rank=0, *args, **kwargs):
        self.hook_before_data_batch()
        r = self._get_a_data_batch(rank, *args, **kwargs)
        rh = self.hook_after_data_batch(r)
        if rh is not None:
            return rh
        else:
            return r

    def one_step(self, rank=0, *args, **kwargs):
        self.hook_before_batch()
        r = self._one_step(rank, *args, **kwargs)
        rh = self.hook_after_batch(r)
        if rh is not None:
            return rh
        else:
            return r

    def do_a_batch_grads(self, *args, **kwargs):
        self.hook_before_do_a_batch_grads()
        r = self._do_a_batch_grads(*args, **kwargs)
        rh = self.hook_after_do_a_batch_grads(r)
        if rh is not None:
            return rh
        else:
            return r

    def do_a_train_step(self, stp=None):
        self.hook_before_do_a_train_step()
        r = self._do_a_train_step(stp=stp)
        rh = self.hook_after_do_a_train_step(r)
        if rh is not None:
            return rh
        else:
            return r

    # to be implemented


    def _before_train(self, rank=0, *args, **kwargs):
        pass

    def _after_train(self, rank=0, *args, **kwargs):
        pass

    def _before_epoch(self, rank=0, *args, **kwargs):
        pass

    def _after_epoch(self, rank=0, *args, **kwargs):
        pass

    def _get_a_data_batch(self, rank=0, *args, **kwargs):
        pass

    def _one_step(self, rank=0, *args, **kwargs):
        pass

    def _do_a_batch_grads(self, *args, **kwargs):
        pass

    def _do_a_train_step(self, stp=None):
        pass

    # hooks

    def get_hook_func_name(self, func):
        fname = str(id(func)) + str(uuid.uuid4())
        return fname

    # NOTE: when self-define a hook func, if hook-befor func, the first arg
    # is the mdwrp instance which is added automatically here; if hook-after
    # func, the first two args are mdwrp and the return of the real func,
    # and the return of hook will replace the real return if it is not None

    def hook_iter_run_before(self, funcs, *args, **kwargs):
        # TODO: get the inputs of real func
        ret = None
        #rettp = args  # the return of the real func
        for fname, fns in funcs.items():
            func = fns[0]
            fargs = fns[1]
            fkwargs = fns[2]
            #fargs = (self,) + rettp + fargs
            fargs = (self,) + fargs
            fkwargs.update(kwargs)
            ret = func(*fargs, **fkwargs)
            #if ret is not None:
            #    rettp = (ret,)
        return ret

    def hook_iter_run_after(self, funcs, *args, **kwargs):
        ret = None
        rettp = args  # the return of the real func
        for fname, fns in funcs.items():
            func = fns[0]
            fargs = fns[1]
            fkwargs = fns[2]
            fargs = (self,) + rettp + fargs
            fkwargs.update(kwargs)
            ret = func(*fargs, **fkwargs)
            if ret is not None:
                rettp = (ret,)
        return ret


    def hook_before_train(self, *args, **kwargs):
        ret = self.hook_iter_run_before(
            self.hook_before_train_funcs, *args, **kwargs)
        return ret

    def hook_after_train(self, *args, **kwargs):
        ret = self.hook_iter_run_after(
            self.hook_after_train_funcs, *args, **kwargs)
        return ret

    def register_hook_before_train(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_before_train_funcs[fname] = [func, args, kwargs]

    def register_hook_after_train(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_after_train_funcs[fname] = [func, args, kwargs]


    def hook_before_epoch(self, *args, **kwargs):
        ret = self.hook_iter_run_before(
            self.hook_before_epoch_funcs, *args, **kwargs)
        return ret

    def hook_after_epoch(self, *args, **kwargs):
        ret = self.hook_iter_run_after(
            self.hook_after_epoch_funcs, *args, **kwargs)
        return ret

    def register_hook_before_epoch(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_before_epoch_funcs[fname] = [func, args, kwargs]

    def register_hook_after_epoch(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_after_epoch_funcs[fname] = [func, args, kwargs]


    def hook_before_data_batch(self, *args, **kwargs):
        ret = self.hook_iter_run_before(
            self.hook_before_data_batch_funcs, *args, **kwargs)
        return ret

    def hook_after_data_batch(self, *args, **kwargs):
        ret = self.hook_iter_run_after(
            self.hook_after_data_batch_funcs, *args, **kwargs)
        return ret

    def register_hook_before_data_batch(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_before_data_batch_funcs[fname] = [func, args, kwargs]

    def register_hook_after_data_batch(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_after_data_batch_funcs[fname] = [func, args, kwargs]


    def hook_before_batch(self, *args, **kwargs):
        ret = self.hook_iter_run_before(
            self.hook_before_batch_funcs, *args, **kwargs)
        return ret

    def hook_after_batch(self, *args, **kwargs):
        ret = self.hook_iter_run_after(
            self.hook_after_batch_funcs, *args, **kwargs)
        return ret

    def register_hook_before_batch(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_before_batch_funcs[fname] = [func, args, kwargs]

    def register_hook_after_batch(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_after_batch_funcs[fname] = [func, args, kwargs]


    def hook_before_do_a_batch_grads(self, *args, **kwargs):
        ret = self.hook_iter_run_before(
            self.hook_before_do_a_batch_grads_funcs, *args, **kwargs)
        return ret

    def hook_after_do_a_batch_grads(self, *args, **kwargs):
        ret = self.hook_iter_run_after(
            self.hook_after_do_a_batch_grads_funcs, *args, **kwargs)
        return ret

    def register_hook_before_do_a_batch_grads(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_before_do_a_batch_grads_funcs[fname] = [func, args, kwargs]

    def register_hook_after_do_a_batch_grads(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_after_do_a_batch_grads_funcs[fname] = [func, args, kwargs]


    def hook_before_do_a_train_step(self, *args, **kwargs):
        ret = self.hook_iter_run_before(
            self.hook_before_do_a_train_step_funcs, *args, **kwargs)
        return ret

    def hook_after_do_a_train_step(self, *args, **kwargs):
        ret = self.hook_iter_run_after(
            self.hook_after_do_a_train_step_funcs, *args, **kwargs)
        return ret

    def register_hook_before_do_a_train_step(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_before_do_a_train_step_funcs[fname] = [func, args, kwargs]

    def register_hook_after_do_a_train_step(self, func, *args, **kwargs):
        fname = self.get_hook_func_name(func)
        self.hook_after_do_a_train_step_funcs[fname] = [func, args, kwargs]


class TrainerOPBase(ModelWrapperHooks):
    def __init__(self, cfg={}):
        super(TrainerOPBase, self).__init__(cfg)

    # -------------------------------------------------------------
    # init
    # TODO:

    # -------------------------------------------------------------
    # nccl

    def cre_nccl(self, msg, *args, **kwargs):
        kn = self.kn
        rank = self.rank
        gpu_i = self.gpu_i

        self.nc = ncclwrp.NcclWrp(kn, rank, gpu_i)

    def cre_nccl_nuid(self, msg, *args, **kwargs):
        nuid = self.nc.get_nuid()
        return nuid

    def init_nccl_comm(self, msg, *args, **kwargs):
        nuid = msg.get('nuid')
        self.nc.set_nuid(nuid)
        self.nc.init_comm()

    def del_nccl_comm(self, msg=None, *args, **kwargs):
        self.nc.del_comm()

    def nccl_stream_sync(self, *args, **kwargs):
        self.nc.stream_sync()


    def do_params_nccl_all_reduce(self, *args, **kwargs):
        pass

    def do_grads_nccl_all_reduce(self, *args, **kwargs):
        pass

    # -------------------------------------------------------------
    # data

    def init_train_data_it(self, *args, **kwargs):
        pass

    def cre_distdt_indices(self, msg, *args, **kwargs):
        pass

    def reset_distdt_indices(self, msg, *args, **kwargs):
        pass

    # -------------------------------------------------------------
    # train

    def check_msg(self, msg, *args, **kwargs):
        pass

    def if_stop(self):
        if self.e > self.epochs:
            return True
        return False

    def clear(self):
        self.clear_train_data()

        self.del_nccl_comm()

        self.after_train()

    def clear_train_data(self, *args, **kwargs):
        pass
