"""PyTorch Trainer for One Process (MP)"""
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
from necklace.frmwrk.pytorch import mputils
from necklace.trainer import tnopbs


class TrainerMPOPPytorch(tnopbs.TrainerOPBase):

    def __init__(self, cfg):
        super(TrainerMPOPPytorch, self).__init__(cfg)

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

        self.net_map = cfg.get('net_map', OrderedDict())  # None
        self.net_map_maxlen = cfg.get('net_map_maxlen')
        if self.net_map_maxlen is None:
            raise Exception('no net_map_maxlen in trainer cfg')
        self.net_map_order = cfg.get('net_map_order', self.rank)

        self.init_cuda()
        self.init_stream()
        self.init_net()
        self.init_inter_trainer()
        self.init_metric()

        # NOTE: do loss with last part for now
        if self.is_last_part():
            self.init_loss()

        self.train_data = None  # NOTE: some nodes will not create dataloader
        self.train_data_it = None
        if self.is_first_part() or self.is_last_part():
            self.init_train_data()
            self.init_train_data_it()

        #self.init_params_recv()
        self.mp_outputs = None  # for forward
        self.mp_inputs = None  # for forward
        self.mp_targets = None  # for backward
        self.mp_grads = None  # for backward
        self.mp_inp_grads = None  # for backward

    def is_first_part(self):
        return self.net_map_order == 0

    def is_last_part(self):
        return self.net_map_order == self.net_map_maxlen - 1

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
        if not isinstance(self.net, (list, tuple, set)):
            submdls = [self.net,]
        else:
            submdls = self.net

        submdl = mputils.ModelPartWrapper(
            submdls=submdls, md_device=self.gpu_i, mp_order=self.net_map_order)
        self.net_ori = self.net
        self.net = submdl

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

    def del_mp_tensors(self, ts):
        if ts is not None:
            for t in ts:
                del t

    def init_mp_inputs(self, shps):
        #if self.mp_inputs is not None:
        #    return

        self.del_mp_tensors(self.mp_inputs)

        ts = ptprms.init_tensors_recv(shps, self.ctx)
        self.mp_inputs = ts

    def init_mp_targets(self, shps):
        #if self.mp_targets is not None:
        #    return

        self.del_mp_tensors(self.mp_targets)

        ts = ptprms.init_tensors_recv(shps, self.ctx, torch.long)
        self.mp_targets = ts

    def init_mp_inp_grads(self, shps):
        #if self.mp_inp_grads is not None:
        #    return

        self.del_mp_tensors(self.mp_inp_grads)

        ts = ptprms.init_tensors_recv(shps, self.ctx)
        self.mp_inp_grads = ts

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
        #if self.use_dist_data_sampler:
        #    # NOTE: to avoid putting batch to index_queue before dataloader reset,
        #    #       here we set the batch_sampler of dataloader to empty
        #    # TODO: ValueError: batch_sampler attribute should not be set after DataLoader is initialized
        #    btch_sampler = sampler.BatchSampler([], 0)
        #    self.train_data.batch_sampler = btch_sampler

        if self.train_data is not None:
            self.train_data_it = iter(self.train_data)

    def clear_train_data(self, *args, **kwargs):
        try:
            self.train_data.clear()
        except Exception as e:
            pass

    def reset_next_epoch(self):
        self.i = 0
        self.i_m = 0
        self.e += 1

    # -------------------------------------------------------------
    # train (rpc server api)

    def train_loop(self, msg, *args, **kwargs):
        pass

    def check_msg(self, msg, *args, **kwargs):

        cmd = msg.get('cmd')

        if cmd == 'weights_sync':
            #self.do_params_nccl_all_reduce()
            #ret = {'ret': 'weights_synced', 'epoch': self.e, 'batch_i': self.i}
            pass  # TODO:

        # ---------------------------------------------------------------------

        elif cmd == 'one_mp_step_start':
            # TODO: do some initiations here

            ret = {
                'ret': 'one_mp_step_started',
                'epoch': self.e, 'batch_i': self.i,
                'net_map_order': self.net_map_order,
            }

        elif cmd == 'one_mp_step_trans_outputs':
            from_rank = msg.get('from_rank')
            to_rank = msg.get('to_rank')

            if to_rank == self.rank:
                input_shps = msg.get('output_shps')
                if (input_shps is None) and (not self.is_first_part()):
                    raise Exception('need input_shps if not the first part')

                self.init_mp_inputs(input_shps)

                self.do_outputs_nccl_p2p_recv(from_rank, to_rank, self.mp_inputs)

            elif from_rank == self.rank:
                self.do_outputs_nccl_p2p_send(from_rank, to_rank, self.mp_outputs)

            ret = {
                'ret': 'one_mp_step_outputs_transed',
                'epoch': self.e, 'batch_i': self.i,
                'net_map_order': self.net_map_order,
            }

        elif cmd == 'one_mp_step_forward':

            inputs, targets = self.get_mp_inputs()

            # save the targets
            if self.is_first_part() and targets is not None:
                if isinstance(targets, (list, tuple, set)):
                    self.mp_targets = targets
                else:
                    self.mp_targets = [targets,]

            if inputs is None:
                ret = {
                    #'ret': 'epoch_end',  # next epoch ===================
                    'ret': 'mp_epoch_end',  # next epoch ===================
                    'epoch': self.e, 'batch_i': self.i,
                    'net_map_order': self.net_map_order,
                }

            else:
                inputs = self.to_ctx(inputs, self.ctx)

                output = self.net(*inputs)  # forward >>>>>>>>>>>>>>>>>>>
                outputs = self.to_list(output)
                self.mp_outputs = outputs  # for trans_outputs
                output_shps = [out.shape for out in outputs]

                if self.mp_targets is not None:
                    target_shps = [tgt.shape for tgt in self.mp_targets]
                else:
                    target_shps = None

                ret = {
                    'ret': 'one_mp_step_forwarded',
                    'epoch': self.e, 'batch_i': self.i,
                    'net_map_order': self.net_map_order,
                    'output_shps': output_shps,
                    'target_shps': target_shps,
                }

        elif cmd == 'one_mp_step_calc_loss':
            from_rank = msg.get('from_rank')
            to_rank = msg.get('to_rank')

            # p2p trans targets from first part to last part
            if to_rank == self.rank:
                target_shps = msg.get('target_shps')
                if (target_shps is None) or (not self.is_last_part()):
                    raise Exception('need target_shps on the last part')

                self.init_mp_targets(target_shps)

                self.do_outputs_nccl_p2p_recv(from_rank, to_rank, self.mp_targets, dtype='long')

            elif from_rank == self.rank:
                self.do_outputs_nccl_p2p_send(from_rank, to_rank, self.mp_targets, dtype='long')

            # do loss.backward and collect grads of last part
            if self.is_last_part():  # to_rank == self.rank
                self._do_a_batch_loss([self.mp_outputs,], [self.mp_targets,], [self.loss,])

                grads = self.net.get_input_grads()  # get grads ################
                self.mp_grads = grads
                grads_shps = [grd.shape for grd in grads]

                # NOTE: do a opt step on last part here
                self.do_a_train_step()
                self.i += 1

            else:
                grads_shps = None

            ret = {
                'ret': 'one_mp_step_loss_calced',
                'epoch': self.e, 'batch_i': self.i,
                'net_map_order': self.net_map_order,
                'grads_shps': grads_shps,
            }

        elif cmd == 'one_mp_step_trans_grads':
            from_rank = msg.get('from_rank')
            to_rank = msg.get('to_rank')

            if to_rank == self.rank:
                grads_shps = msg.get('grads_shps')
                if (grads_shps is None) and (not self.is_last_part()):
                    raise Exception('need grads_shps if not the last part')

                self.init_mp_inp_grads(grads_shps)

                self.do_outputs_nccl_p2p_recv(from_rank, to_rank, self.mp_inp_grads)

            elif from_rank == self.rank:
                self.do_outputs_nccl_p2p_send(from_rank, to_rank, self.mp_grads)

            ret = {
                'ret': 'one_mp_step_grads_transed',
                'epoch': self.e, 'batch_i': self.i,
                'net_map_order': self.net_map_order,
            }

        elif cmd == 'one_mp_step_backward':

            self.net.mp_backward(self.mp_inp_grads)  # backward <<<<<<<<<<<<

            if not self.is_first_part():  # NOTE: first part no need input grads
                grads = self.net.get_input_grads()  # get grads ################
                self.mp_grads = grads
                grads_shps = [grd.shape for grd in grads]
            else:
                grads_shps = None

            self.do_a_train_step()
            self.i += 1

            ret = {
                'ret': 'one_mp_step_backwarded',
                'epoch': self.e, 'batch_i': self.i,
                'net_map_order': self.net_map_order,
                'grads_shps': grads_shps,
            }

        # ---------------------------------------------------------------------

        elif cmd == 'next_epoch':
            self.after_epoch(self.rank)

            self.reset_next_epoch()

            self.init_train_data_it()

            self.next_epoch_flag = False

            ret = {'ret': 'next_epoch', 'epoch': self.e}

        else:
            ret = {'ret': 'wrong_cmd', 'epoch': self.e}

        return ret


    def get_mp_inputs(self):
        if self.is_first_part():
            dt = self.get_a_data_batch(self.rank)
            if dt is None:
                x = None
                y = None
            else:
                x, y = dt  # NOTE: maybe the x, y each is a list

        else:
            if not self.mp_inputs:
                raise Exception('not trans_outputs for inputs')
            x = self.mp_inputs
            y = None

        return x, y

    # =============================================================

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

    # NOTE: no hooks for now
    def _do_a_batch_loss(self, outputs, targets, loss_fns, *args, **kwargs):
        output = outputs[0]
        target = targets[0]
        loss = loss_fns[0]

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

    # =============================================================

    def cuda_sync(self, *args, **kwargs):
        self.default_stream.synchronize()
        #self.nc.stream_sync()
        torch.cuda.synchronize()

    def do_params_nccl_all_reduce(self, *args, **kwargs):
        self.cuda_sync()
        ptprms.params_do_nccl_allreduce(
            None, self.kn, self.net, self.prm_recvs)
        self.cuda_sync()

    def do_grads_nccl_all_reduce(self, *args, **kwargs):
        self.cuda_sync()
        ptprms.opt_grads_do_nccl_allreduce(
            None, self.kn, self.opt, self.prm_recvs,
            grad_comp_cfg=self.grad_comp_cfg)
        self.cuda_sync()

    def do_outputs_nccl_p2p_send(self, rank_send, rank_recv, send_tensors, dtype='float', *args, **kwargs):
        self.cuda_sync()
        ptprms.tensors_do_nccl_p2p_send(
            self.nc, rank_send, rank_recv, send_tensors, dtype=dtype)
        self.cuda_sync()

    def do_outputs_nccl_p2p_recv(self, rank_send, rank_recv, recv_tensors, dtype='float', *args, **kwargs):
        self.cuda_sync()
        ptprms.tensors_do_nccl_p2p_recv(
            self.nc, rank_send, rank_recv, recv_tensors, dtype=dtype)
        self.cuda_sync()

