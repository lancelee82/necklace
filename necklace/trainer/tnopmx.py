"""MXNet Gluon Trainer for One Process"""
from __future__ import print_function, absolute_import, division

import copy
import random
import time
from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray

from necklace.cuda import nbutils
from necklace.cuda import ncclwrp
from necklace.frmwrk.mxnet import metric as my_metric
from necklace.frmwrk.mxnet import mxprms
from necklace.trainer import trnrmx as my_trainer


class TrainerOPGluon(object):

    def __init__(self, cfg):
        self.cfg = cfg

        self.kn = cfg.get('kn')
        self.ti = cfg.get('ti')
        self.q_msg = cfg.get('q_msg')
        self.q_data = cfg.get('q_data')
        self.q_ctl = cfg.get('q_ctl')
        self.ctx = cfg.get('ctx')
        self.net = cfg.get('net')
        self.net_init = cfg.get('net_init')
        self.opt = cfg.get('opt')
        self.mtrc = cfg.get('mtrc')
        self.optimizer_params = cfg.get('optimizer_params')
        self.batch_size = cfg.get('batch_size')
        self.epochs = cfg.get('epochs')
        self.dataloader_creator = cfg.get('dataloader_creator')
        self.dataloader_creator_args = cfg.get('dataloader_creator_args')
        self.log_interval = cfg.get('log_interval')
        self.grad_comp_cfg = cfg.get('grad_comp_cfg')

        self.rank = self.ti

        self.n = 0  # TODO: batch n
        self.i = 0  # batch i
        self.e = 0  # epoch

        self.next_epoch_flag = False

        self.btic = time.time()

        self.init_cuda()
        self.init_net()
        self.init_inter_trainer()
        self.init_metric()
        self.init_loss()  # TODO: loss as args
        self.init_train_data()
        self.init_train_data_it()
        self.init_params_recv()
        self.init_grads_comp()

    def init_cuda(self, *args, **kwargs):
        if isinstance(self.ctx, mx.Context):
            self.ctx = [self.ctx]

        # NOTE: gpu index on this node, sometimes is different with self.rank
        self.gpu_i = int(self.ctx[0].device_id) #self.ctx[0].device_type

        # NOTE: we do not select gpu device here now, and do this at
        # user application as early as we get the gpu device id
        #nbutils.cuda_select_device(self.gpu_i)

    def init_net(self, msg=None, *args, **kwargs):
        self.net.initialize(self.net_init, ctx=self.ctx)

    def init_inter_trainer(self, msg=None, *args, **kwargs):

        #kv = mx.kv.create(args.kv_store)
        kv = None  # NOTE: no kv -> not sync data on contexts

        comp_params = None
        #comp_params = {'type':'2bit', 'threshold':0.5}
        #comp_params = {'type':'2bit', 'threshold':5.0}
        #trainer = gluon.Trainer(net.collect_params(), 'sgd',
        #                        {'learning_rate': args.lr,
        #                         'wd': args.wd, 'momentum': args.momentum},
        #                        kvstore = kv,
        #                        compression_params=comp_params)
        #trainer = my_trainer.AsyncTrainer(
        self.trainer = my_trainer.SyncTrainer(
            self.net.collect_params(),
            self.opt, #'sgd',
            self.optimizer_params,
            kvstore = kv,
            compression_params=comp_params)

    def init_metric(self, msg=None, *args, **kwargs):

        #self.metric = mx.metric.Accuracy()
        #self.metric = my_metric.Accuracy()
        #self.metric = my_metric.DoNothingMetric()
        #self.metric = my_metric.GPUEvalMetric(ctx[0])
        self.metric = self.mtrc

    def init_loss(self, msg=None, *args, **kwargs):  # TODO: loss as args
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()

    def init_train_data(self, *args, **kwargs):
        # TODO: outside here, split the data
        self.train_data = self.dataloader_creator(
            *self.dataloader_creator_args, ti=self.ti)

    def init_train_data_it(self, *args, **kwargs):
        self.train_data_it = iter(self.train_data)

    def clear_train_data(self, *args, **kwargs):
        try:
            self.train_data.clear()
        except Exception as e:
            pass

    def init_params_recv(self):
        # really init params
        self.one_step()

        self.prm_recvs = mxprms.init_params_recvs(self.net.collect_params())

    def init_grads_comp(self):
        if self.grad_comp_cfg:
            grd_siz_thrd = self.grad_comp_cfg.get('grad_size_threshold', 1001)
            grd_cmp_thrd = self.grad_comp_cfg.get('grad_comp_threshold', 0.5)

            self.grd_resi, self.grd_recv, self.grd_comp, self.grd_cmps = \
                mxprms.init_grads_compression(
                    self.net.collect_params(), self.kn,
                    grd_siz_thrd=grd_siz_thrd)

            self.grad_comp_cfg['grd_resi'] = self.grd_resi
            self.grad_comp_cfg['grd_recv'] = self.grd_recv
            self.grad_comp_cfg['grd_comp'] = self.grd_comp
            self.grad_comp_cfg['grd_cmps'] = self.grd_cmps

            self.grad_comp_cfg['grad_size_threshold'] = grd_siz_thrd
            self.grad_comp_cfg['grad_comp_threshold'] = grd_cmp_thrd


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

    # -------------------------------------------------------------
    # train

    def train_loop(self, msg, *args, **kwargs):
        pass

    def check_msg(self, msg, *args, **kwargs):

        cmd = msg.get('cmd')
        typ = msg.get('typ')  # 1: weights  2: gradients

        if cmd == 'weights_sync':
            self.do_params_nccl_all_reduce()
            ret = {'ret': 'weights_synced', 'epoch': self.e, 'batch_i': self.i}

        elif cmd == 'one_step':
            if not self.next_epoch_flag:
                r = self.one_step(msg)
            else:
                r = 'epoch_end'

            if r == 'one_step':
                if typ == 'weig':
                    self.do_a_train_step()

            if r == 'epoch_end':
                self.next_epoch_flag = True

            ret = {'ret': r, 'epoch': self.e}

        elif cmd == 'nccl_all_reduce':
            if typ == 'grad':
                self.do_grads_nccl_all_reduce()
                self.nccl_stream_sync()

                self.do_a_train_step()

            #elif typ == 'weig':
            else:
                self.do_params_nccl_all_reduce()

            ret = {'ret': 'nccl_all_reduced', 'epoch': self.e, 'batch_i': self.i}

        elif cmd == 'next_epoch':
            self.init_train_data_it()
            self.next_epoch_flag = False

            ret = {'ret': 'next_epoch', 'epoch': self.e}

        else:
            ret = {'ret': 'wrong_cmd', 'epoch': self.e}

        # TODO:
        #self.metric_log()

        return ret

    def one_step(self, msg=None, *args, **kwargs):
        dt = self._get_a_data_batch(self.rank)
        if not dt:
            self.i = 0
            self.e += 1
            ret = 'epoch_end'
        else:
            x, y = dt
            self.do_a_batch_grads([x], [y], [self.loss])

            self.i += 1
            ret = 'one_step'

        # TODO:
        self.metric_log()

        return ret

    def _get_a_data_batch(self, rank=0, *args, **kwargs):
        try:
            dt = next(self.train_data_it)
            return dt
        except StopIteration as e:
            return None
        except Exception as e:
            return None

    def do_a_batch_grads__1(self, inputs, targets, loss_fns, *args, **kwargs):
        data = inputs[0]
        label = targets[0]
        loss = loss_fns[0]

        with autograd.record():
            output = self.net(data)
            L = loss(output, label)
            L.backward()

        self.trainer.step(data.shape[0])

        # update metric at last
        if self.metric:
            if self.i == 0:  # a new epoch
                self.metric.reset()
            self.metric.update([label], [output])

    def do_a_batch_grads(self, inputs, targets, loss_fns, *args, **kwargs):
        data = inputs[0]
        label = targets[0]
        loss = loss_fns[0]

        data = gluon.utils.split_and_load(data, ctx_list=self.ctx, batch_axis=0)
        label = gluon.utils.split_and_load(label, ctx_list=self.ctx, batch_axis=0)
        outputs = []
        Ls = []
        with autograd.record():
            for x, y in zip(data, label):
                z = self.net(x)
                L = loss(z, y)
                # store the loss and do backward after we have done forward
                # on all GPUs for better speed on multiple GPUs.
                Ls.append(L)
                outputs.append(z)
        for L in Ls:
            L.backward()

        if self.metric:
            if self.i == 0:  # a new epoch
                self.metric.reset()
            self.metric.update(label, outputs)

    def do_a_train_step(self, stp=None):

        # NOTE: wait the weights
        ndarray.waitall()

        if not stp:
            stp = self.batch_size
        self.trainer.step(stp)

        # TODO: zero grad
        mxprms.params_zero_grad(self.net.collect_params())

        # for debug
        #mxprms.params_weights_show(self.net.collect_params())

    def opt_get_lr(self):
        lr = self.optimizer_params.get('learning_rate', 0.01)
        return lr

    def do_a_train_step__2x(self, stp=None):
        lr = self.opt_get_lr()
        params = self.net.collect_params()
        mxprms.params_opt_update(params, lr)

        # TODO: zero grad
        mxprms.params_zero_grad(params)

    def do_params_nccl_all_reduce(self, *args, **kwargs):
        mxprms.params_do_nccl(self.net.collect_params(),
                              self.prm_recvs, self.nc, self.kn)

    def do_grads_nccl_all_reduce(self, *args, **kwargs):
        mxprms.grads_do_nccl(self.net.collect_params(),
                             self.prm_recvs, self.nc, self.kn,
                             grad_comp_cfg=self.grad_comp_cfg)

    def nccl_stream_sync(self, *args, **kwargs):
        self.nc.stream_sync()

    def metric_log(self, *args, **kwargs):
        if self.log_interval and (self.i + 1) % self.log_interval == 0:
            name, acc = self.metric.get()
            batch_size = self.batch_size * self.kn * self.log_interval
            print(batch_size, (time.time() - self.btic))
            print('[%d]Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                   self.ti, self.e, self.i,
                   batch_size/(time.time() - self.btic), name, acc))

            self.btic = time.time()

    def check_if_stop(self):
        if self.e >= self.epochs:
            return True
        else:
            return False

    def clear(self):
        self.clear_train_data()

        self.del_nccl_comm()
