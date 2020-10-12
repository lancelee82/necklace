"""PyTorch Trainer for One Process (DP)"""
import six
import copy
import random
import time
from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from necklace.cuda import nbutils
from necklace.cuda import ncclwrp
from necklace.data import sampler
from necklace.data import dtdist
#from necklace.frmwrk.pytorch import metric as my_metric
from necklace.frmwrk.pytorch import ptprms
from necklace.frmwrk.pytorch import ptutils
from necklace.trainer import tnopbs


class TrainerOPPytorch(tnopbs.TrainerOPBase):

    def __init__(self, cfg):
        super(TrainerOPPytorch, self).__init__(cfg)

        self.kn = cfg.get('kn')
        self.ti = cfg.get('ti')
        self.ctx = cfg.get('ctx')
        self.net = cfg.get('net')
        self.net_init = cfg.get('net_init')  # None
        self.opt = cfg.get('opt')
        self.loss = cfg.get('loss')
        self.mtrc = cfg.get('mtrc')
        self.optimizer_params = cfg.get('optimizer_params')
        self.batch_size = cfg.get('batch_size')
        self.epochs = cfg.get('epochs')
        self.dataloader_creator = cfg.get('dataloader_creator')
        self.dataloader_creator_args = cfg.get('dataloader_creator_args', ())
        self.dataloader_creator_kwargs = cfg.get('dataloader_creator_kwargs', {})
        self.use_dist_data_sampler = cfg.get('use_dist_data_sampler', False)  # TODO: always False now, to delete
        self.log_interval = cfg.get('log_interval')
        self.grad_comp_cfg = cfg.get('grad_comp_cfg')

        self.my_kwargs = cfg.get('my_kwargs', {})  # used for saving some global vars

        self.rank = self.ti

        self.n = 0  # TODO: batch n
        self.i = 0  # batch i
        self.i_m = 0  # batch i for metric log
        self.e = 0  # epoch

        self.next_epoch_flag = False

        self.dt_sampler_indices = None

        self.btic = time.time()

        self.init_cuda()
        self.init_stream()
        self.init_net()
        self.init_inter_trainer()
        self.init_metric()
        self.init_loss()
        self.init_params_recv()
        self.init_train_data()
        self.init_train_data_it()
        #self.init_grads_comp()  # TODO:

    def init_cuda(self, *args, **kwargs):
        # NOTE: gpu index on this node, sometimes is different with self.rank
        if isinstance(self.ctx, torch.device):
            self.gpu_i = int(self.ctx.index)
        elif isinstance(self.ctx, six.integer_types):
            self.gpu_i = self.ctx
            self.ctx = torch.device(self.gpu_i)
        else:
            raise Exception('the ctx should be an int or torch.device')

        # NOTE: we do not select gpu device here now, and do this at
        # user application as early as we get the gpu device id with
        #nbutils.cuda_select_device(self.gpu_i)

    def init_stream(self):
        self.default_stream = torch.cuda.current_stream()
        self.nccl_stream = torch.cuda.Stream()

    def init_net(self, msg=None, *args, **kwargs):
        self.net.to(self.ctx)
        self.net.train()  # TODO: do at a better place ?

    def init_inter_trainer(self, msg=None, *args, **kwargs):
        pass

    def init_metric(self, msg=None, *args, **kwargs):
        self.metric = self.mtrc

    def init_loss(self, msg=None, *args, **kwargs):
        if not self.loss:
            self.loss = F.nll_loss
        self.lss = None

    def init_params_recv(self):  # TODO: use in-place
        self.prm_recvs = ptprms.init_params_recvs(self.net)

    # -------------------------------------------------------------
    # data loader

    def init_train_data(self, *args, **kwargs):
        # TODO: outside here, split the data

        # NOTE: increase the interval time for waiting for the dt_sampler_sync
        import torch.utils.data
        torch.utils.data.dataloader.MANAGER_STATUS_CHECK_INTERVAL = 10. * self.kn

        self.train_data = self.dataloader_creator(
            *self.dataloader_creator_args,
            **self.dataloader_creator_kwargs)

        self.n = len(self.train_data)

        self.ds_len = len(self.train_data.dataset)  # the len of the dataset
        self.batch_size = self.train_data.batch_size

    def init_train_data_it(self, *args, **kwargs):
        if self.use_dist_data_sampler:
            # NOTE: to avoid putting batch to index_queue before dataloader reset,
            #       here we set the batch_sampler of dataloader to empty
            # TODO: ValueError: batch_sampler attribute should not be set after DataLoader is initialized
            btch_sampler = sampler.BatchSampler([], 0)
            self.train_data.batch_sampler = btch_sampler

        self.train_data_it = iter(self.train_data)

    def clear_train_data(self, *args, **kwargs):
        try:
            self.train_data.clear()
        except Exception as e:
            pass

    # -------------------------------------------------------------
    # distributed dataloader sampler (not used now)

    def init_data_sampler(self, dn):
        shuffle = self.dataloader_creator_kwargs.get('shuffle', False)
        ssmp = sampler.SliceSampler(dn, shuffle=shuffle)
        s = torch.IntTensor(ssmp.indices)
        return s

    def do_nccl_bcast_dt_sampler(self, ssmp, root=0):
        self.cuda_sync()
        '''
        # TODO: [not tested] maybe this is no need
        with torch.cuda.stream(self.nccl_stream):
            s = dtdist.dt_sampler_do_nccl_bcast(self.nc, self.kn, root, ssmp)
            #s = dtdist.dt_sampler_do_nccl_allreduce(self.nc, self.kn, root, ssmp)
        self.default_stream.wait_stream(self.nccl_stream)
        '''
        s = dtdist.dt_sampler_do_nccl_allreduce(self.nc, self.kn, root, ssmp)
        self.cuda_sync()
        return s

    def sync_dist_data_sampler(self):
        dn = self.ds_len
        batch_size = self.batch_size

        if self.rank == 0:
            #s = self.init_data_sampler(dn)
            a = list(range(dn))
            random.shuffle(a)
            s = torch.IntTensor(a)
        else:
            #s = torch.IntTensor(range(dn))
            s = torch.IntTensor([0] * dn)

        s = s.cuda(self.ctx)  # NOTE: NCCL ask for buf on gpu
        #print('s 2', s)
        _x_ = repr(s)  # NOTE[!!!]: this is necessary for cuda stream sync
        r = self.do_nccl_bcast_dt_sampler(s)
        r = r.cpu().numpy()

        n = len(r)
        # make sure all workers have the same number of batches
        b = n // (batch_size * self.kn)
        indices = r[batch_size*b*self.rank:batch_size*b*(self.rank+1)]
        self.dt_sampler_indices = indices

    def reset_data_loader_pth(self):
        batch_size = self.batch_size
        btch_sampler = sampler.BatchSampler(
            self.dt_sampler_indices, batch_size, last_batch='discard')

        # update the batch_sampler of the dataloader
        self.train_data.batch_sampler = btch_sampler
        self.train_data_it.batch_sampler = btch_sampler
        self.train_data_it.sample_iter = iter(btch_sampler)

        # prime the prefetch loop
        # (copy from torch.utils.data.dataloader._DataLoaderIter)
        for _ in range(2 * self.train_data.num_workers):
            self.train_data_it._put_indices()

    def reset_data_loader_ngn(self):
        slc_sampler = sampler.SliceSampler(indices=self.dt_sampler_indices)
        self.train_data.reset_batch_sampler(slc_sampler)

    def reset_data_loader(self):
        if hasattr(self.train_data, '_attr_ngn_dataloader'):
            self.reset_data_loader_ngn()
        else:
            self.reset_data_loader_pth()

        self.n = len(self.train_data)

    # -------------------------------------------------------------
    # distributed dataloader sampler indices slicer (not used now)
    # NOTE: use_dist_data_sampler is always False

    def cre_distdt_indices(self, msg, *args, **kwargs):
        dt_indices = np.arange(self.ds_len)
        shuffle = self.dataloader_creator_kwargs.get('shuffle', False)
        if shuffle:
            random.shuffle(dt_indices)
        return dt_indices

    def reset_distdt_indices(self, msg, *args, **kwargs):
        r = msg.get('dt_indices')

        if not self.use_dist_data_sampler:
            return

        batch_size = self.batch_size
        n = len(r)
        # make sure all workers have the same number of batches
        b = n // (batch_size * self.kn)
        indices = r[batch_size*b*self.rank:batch_size*b*(self.rank+1)]
        self.dt_sampler_indices = indices

        # TODO: ValueError: batch_sampler attribute should not be set after DataLoader is initialized
        self.reset_data_loader()

    # -------------------------------------------------------------
    # train (rpc server api)

    def train_loop(self, msg, *args, **kwargs):
        pass

    def check_msg(self, msg, *args, **kwargs):

        cmd = msg.get('cmd')
        typ = msg.get('typ')  # 1: weig  2: grad

        if cmd == 'weights_sync':
            self.do_params_nccl_all_reduce()
            ret = {'ret': 'weights_synced', 'epoch': self.e, 'batch_i': self.i}

        elif cmd == 'dt_sampler_sync':  # TODO: not used now
            #if self.use_dist_data_sampler:
            #    self.sync_dist_data_sampler()
            #    self.reset_data_loader()

            ret = {'ret': 'dt_sampler_synced', 'epoch': self.e, 'batch_i': self.i}

        elif cmd == 'one_step':
            if not self.next_epoch_flag:
                r = self.one_step(msg)
            else:
                r = 'epoch_end'

            if r == 'one_step':
                if typ == 'weig':
                    # <2.1> if allreduce weights, do a optimizer step after a forward-backward immediately
                    self.do_a_train_step()

            if r == 'epoch_end':
                self.next_epoch_flag = True

            ret = {'ret': r, 'epoch': self.e}

        elif cmd == 'nccl_all_reduce':
            if typ == 'grad':
                # <3.1> if allreduce gradients, do grads allreduce at first
                self.do_grads_nccl_all_reduce()
                self.nccl_stream_sync()

                # <3.2> if allreduce gradients, do optimizer step after the grads allreduce
                self.do_a_train_step()

            #elif typ == 'weig':
            else:
                # <2.2> if allreduce weights, do weights allreduce after optimezer step
                self.do_params_nccl_all_reduce()

            ret = {'ret': 'nccl_all_reduced', 'epoch': self.e, 'batch_i': self.i}

        elif cmd == 'next_epoch':
            self.after_epoch(self.rank)

            self.init_train_data_it()

            self.next_epoch_flag = False

            ret = {'ret': 'next_epoch', 'epoch': self.e}

        else:
            ret = {'ret': 'wrong_cmd', 'epoch': self.e}

        return ret

    def _one_step(self, rank=0, *args, **kwargs):
        dt = self.get_a_data_batch(self.rank)

        if not dt:
            self.i = 0
            self.i_m = 0
            self.e += 1
            ret = 'epoch_end'
        else:
            x, y = dt  # NOTE: maybe the x, y each is a list

            self.do_a_batch_grads([x], [y], [self.loss])

            self.i += 1
            ret = 'one_step'

            # TODO: to a callback
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

    def _do_a_batch_grads(self, inputs, targets, loss_fns, *args, **kwargs):
        data = inputs[0]
        target = targets[0]
        loss = loss_fns[0]

        data = self.to_ctx(data, self.ctx)
        target = self.to_ctx(target, self.ctx)

        if isinstance(self.opt, (list, tuple)):
            for opt in self.opt:
                opt.zero_grad()
        else:
            self.opt.zero_grad()

        # forward
        output = self.net(*data)

        output = self.to_list(output)
        self.lss = loss(*output, *target)
        # TODO: or use following format and use loss wrapper
        # NOTE: the output and target both may be list or tuple
        #self.lss = loss(output, target)

        if isinstance(self.lss, (list, tuple)):
            for li, l in enumerate(self.lss):
                if li == len(self.lss) - 1:
                    l.backward()
                else:
                    l.backward(retain_graph=True)  # NOTE: need to retain grad
        else:
            self.lss.backward()

        if self.metric:
            if self.i == 0:  # a new epoch
                self.metric.reset()
            self.metric.update(*target, *output)

    def _do_a_batch_loss(self, outputs, targets, loss_fns, *args, **kwargs):
        pass

    def _do_a_train_step(self, stp=None):
        if isinstance(self.opt, (list, tuple)):
            for opt in self.opt:
                opt.step()
        else:
            self.opt.step()

    # -------------------------------------------------------------
    # train (util functions)

    def to_ctx(self, dt, ctx):
        if isinstance(dt, (list, tuple)):
            r = []
            for d in dt:
                c = self.try_to_ctx(d, ctx)
                r.append(c)
        else:
            r = [self.try_to_ctx(dt, ctx)]  # NOTE: this is a list
        return r

    def try_to_ctx(self, d, ctx):
        if isinstance(d, (torch.Tensor,)):
            return d.to(ctx)
        else:
            return d
            #x#return self.to_ctx(d, ctx)

    def to_list(self, x):
        if isinstance(x, (list, tuple)):
            pass
        else:
            x = [x]
        return x

    def do_params_nccl_all_reduce(self, *args, **kwargs):
        self.cuda_sync()
        ptprms.params_do_nccl_allreduce(
            self.nc, self.kn, self.net, self.prm_recvs)
        self.cuda_sync()

    def do_grads_nccl_all_reduce(self, *args, **kwargs):
        self.cuda_sync()
        #ptprms.grads_do_nccl_allreduce(
        #    self.nc, self.kn, self.net, self.prm_recvs,
        #    grad_comp_cfg=self.grad_comp_cfg)
        # NOTE: use parameters in optimeter(s)
        ptprms.opt_grads_do_nccl_allreduce(
            self.nc, self.kn, self.opt, self.prm_recvs,
            grad_comp_cfg=self.grad_comp_cfg)
        self.cuda_sync()

    def cuda_sync(self, *args, **kwargs):
        self.default_stream.synchronize()
        #self.nc.stream_sync()
        torch.cuda.synchronize()

    def metric_log(self, *args, **kwargs):  # TODO: to a callback
        if (self.log_interval and (self.i + 1) % self.log_interval == 0) or \
                self.check_if_last_batch():

            bn = self.i - self.i_m
            self.i_m = self.i
            batch_size = self.batch_size * self.kn * bn
            batch_time = (time.time() - self.btic)
            print(batch_size, batch_time)

            if self.metric:
                name, acc = self.metric.get()
                info = '%s=%f' % (name, acc)
            else:
                if isinstance(self.lss, (list, tuple)):
                    info = ''
                    for l in self.lss:
                        info += ('%f ' % (l.item(),))
                else:
                    name, acc = 'loss', self.lss.item()
                    info = '%s=%f' % (name, acc)

            print('[%d]Epoch[%d] Batch [%d / %d]\tSpeed: %f samples/sec \t%s' % (
                   self.ti, self.e, self.i, self.n,
                   batch_size / batch_time, info))

            self.btic = time.time()

    def check_if_last_batch(self):
        dn = len(self.train_data)
        return dn == self.i
