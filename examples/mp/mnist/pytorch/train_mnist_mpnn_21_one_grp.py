from __future__ import division

import os
import sys
import argparse
import time
import logging
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURR_DIR))))  # ../../../../
sys.path.insert(0, ROOT_DIR)

import necklace
from necklace.trainer import tndsnc
from necklace.rpc import ilog
from necklace.utils import hookbase
from necklace.utils import wrpbase
from necklace.utils.argutils import attach_nk_args_parser


# CLI
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch-size', '-b', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

attach_nk_args_parser(parser)

args = parser.parse_args()


from necklace.cuda import ncclgrp
#ncclgrp.set_nccl_group_main_world_size(args.world_size)
#ncclgrp.set_nccl_group_main_rank(args.rank)
pg_cfg = {
    'world_size': args.world_size,
    'my_rank': args.rank,
    'dp_size': args.dp_size,
    'mp_size': args.mp_size,
    #'pp_size': args.pp_size,
    #'zr_size': args.zr_size,
    'is_worker': args.role == 'worker',
}
ncclgrp.init_nccl_groups_map(args.tmode, cfg=pg_cfg)

nccl_groups_cfg_map = ncclgrp.get_nccl_groups_map_cfg_dict()  # only my
nccl_groups_grp_map = ncclgrp.get_nccl_groups_map_grp_dict()  # all grps
nccl_group_main_grp_list = ncclgrp.get_nccl_group_main_grp_list()  # main [[...],]
print('nccl_groups_cfg_map', nccl_groups_cfg_map)
print('nccl_groups_grp_map', nccl_groups_grp_map)
print('nccl_group_main_grp_list', nccl_group_main_grp_list)
#pp_rank = nccl_groups_cfg_map.get('pp_group', {}).get('rank')  # TODO: to global
#print('pp_rank: ', pp_rank)


# ----------------------------------------------------------------------------
# original training code

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------------
from necklace.frmwrk.pytorch.mpnn.initialize import get_model_parallel_world_size
from necklace.frmwrk.pytorch.mpnn.layers import ColumnParallelLinear
from necklace.frmwrk.pytorch.mpnn.layers import RowParallelLinear
from necklace.frmwrk.pytorch.mpnn.mappings import gather_from_model_parallel_region
from necklace.frmwrk.pytorch.mpnn.utils import divide
from necklace.frmwrk.pytorch.mpnn.utils import split_tensor_along_last_dim


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class ParallelDenseReluDense(nn.Module):
    def __init__(self,
                 config=None,
                 init_method=None,
                 output_layer_init_method=None):
        super(ParallelDenseReluDense, self).__init__()
        inp_dim = 28 * 28
        d_model = 40960
        out_dim = 10
        self.wi_0 = ColumnParallelLinear(
            #config.d_model, config.d_ff,
            inp_dim, d_model,
            gather_output=False,
            bias=False,
            #init_method=init_method
        )
        self.wi_1 = ColumnParallelLinear(
            #config.d_model, config.d_ff,
            inp_dim, d_model,
            gather_output=False,
            bias=False,
            #init_method=init_method
        )
        self.wo = RowParallelLinear(
            #config.d_ff, config.d_model,
            d_model, out_dim,
            bias=False,
            input_is_parallel=True,
            #init_method=output_layer_init_method
        )
        self.dropout = nn.Dropout(0.1)#config.dropout_rate)

        # self.do_dim_trick = config.do_dim_trick
        # if torch.distributed.get_rank() % 5 == 4:
        #     self.ff_mask = nn.Parameter(torch.tensor([1.0] * 13104 + [0.0] * 4), requires_grad=False)
        # else:
        #     self.ff_mask = nn.Parameter(torch.tensor([1.0] * 13108), requires_grad=False)

    def forward(self, hidden_states):
        # hidden_states: [b, s, hp]
        B = hidden_states.size(0)
        hidden_states = hidden_states.view(B, -1)
        hidden_gelu = gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        # hidden_states: [b, s, d_ff_p]
        # if self.do_dim_trick:
        #     ff_mask = self.ff_mask.view(1, 1, self.ff_mask.size(0))
        #     hidden_states = ff_mask * hidden_states

        # hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        # hidden_states: [b, s, hp]

        hidden_states = F.log_softmax(hidden_states, dim=1)  # TODO: for nll_loss

        return hidden_states


class MpFc2(nn.Module):
    def __init__(self):
        super(MpFc2, self).__init__()

        inp_dim = 28 * 28
        d_model = 40960
        out_dim = 10

        self.fc1 = ColumnParallelLinear(
            inp_dim, d_model,
            gather_output=False,  # NOTE: do NOT gather outputs, result in [???]
            bias=False,
            #init_method=init_method
        )
        self.fc2 = RowParallelLinear(
            d_model, out_dim,
            #gather_output=False,  # NOTE: default is True
            bias=False,
            input_is_parallel=True,
            #init_method=output_layer_init_method
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)

        o = self.fc1(x)
        o = F.relu(o)
        o = self.fc2(o)

        o = F.log_softmax(o, dim=1)  # TODO: for nll_loss

        return o


class DpFc2(nn.Module):  # not for MP but for DP (for diff)
    def __init__(self):
        super(DpFc2, self).__init__()

        inp_dim = 28 * 28
        d_model = 40960
        out_dim = 10

        self.fc1 = nn.Linear(
            inp_dim, d_model,
            bias=False,
        )
        self.fc2 = nn.Linear(
            d_model, out_dim,
            bias=False,
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)

        o = self.fc1(x)
        o = F.relu(o)
        o = self.fc2(o)

        o = F.log_softmax(o, dim=1)

        return o


# ----------------------------------------------------------------------------
def get_net_ctx(args):
    gpu_dev_i = int(args.gpu_rank)
    ctx = torch.device(gpu_dev_i)

    #model = Net().to(ctx)
    #model = ParallelDenseReluDense().to(ctx)
    model = MpFc2().to(ctx)
    #model = DpFc2().to(ctx)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum)

    return model, optimizer, ctx


def get_data_loader(*args, **kwargs):
    batch_size = kwargs.get('batch_size', 64)

    # NOTE: for distributed sampler/dataloader
    kn = kwargs.get('kn', 1)
    rank = kwargs.get('rank', 0)  # rank
    shuffle = kwargs.get('shuffle', True)

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    # ============================================================
    # NOTE: MP training without DP needs the same input at workers,
    #       so here the shuffle is False;
    #       Another (maybe better) method is using the data_sync
    #       where the rank-0 will broadcast the data to others
    #       in its mp_group
    # ============================================================
    #kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    kwargs = {'num_workers': 1, 'pin_memory': True} #if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, **kwargs)

    test_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': False}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, **test_kwargs)

    return train_loader, test_loader


def get_train_dataloader(*args, **kwargs):
    train_loader, test_loader = get_data_loader(*args, **kwargs)
    return train_loader


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# ----------------------------------------------------------------------------
# necklace hooks and wrappers

class HookDoTest(hookbase.NklcHookBase):
    def __init__(self, aargs, net, ctx, *args, **kwargs):
        super(HookDoTest, self).__init__(net, ctx, *args, **kwargs)
        self.aargs = aargs

    def __call__(self, trnr, *args, **kwargs):
        _, test_loader = get_data_loader(batch_size=self.aargs.batch_size)
        test(self.aargs, self.net, self.ctx, test_loader)


class WrpDoLoss(wrpbase.NklcWrapperBase):
    def __init__(self, net, ctx, loss_fn, *args, **kwargs):
        super(WrpDoLoss, self).__init__(net, ctx, *args, **kwargs)
        self.loss_fn = loss_fn

    def __call__(self, output, target, *args, **kwargs):
        loss = self.loss_fn(output, target)
        print('loss', loss.cpu().item())
        return loss


# ----------------------------------------------------------------------------
#ilog.il_add(500001)
#ilog.il_add(500002)


if __name__ == '__main__':

    role = args.role

    if role == 'scheduler':
        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'wdsz': args.world_size,
            'rank': args.rank,

            'epochs': args.epochs,
            # ========================================================
            # NOTE: we do only opt.step() but no grad_allreduce in tnoppt
            # ========================================================
            'nccl_allreduce_typ': 'oopt', #'grad',  # 'weig'
        }

        svc = tndsnc.SVCScheduler(cfg)
        svc.loop()

    elif role == 'worker':

        gpu_dev_i = int(args.gpu_rank)

        # ========================================================
        # do necklace init (mp/cuda) here only in worker
        # ========================================================
        necklace.init(gpu_dev_i)

        # set cuda device of pytorch
        torch.cuda.set_device(gpu_dev_i)

        from necklace.trainer import tnoppt
        from necklace.frmwrk.pytorch.ptutils import prnt_md

        net, opt, ctx = get_net_ctx(args)
        prnt_md(net)

        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'wdsz': args.world_size,
            'rank': args.rank,

            #'svr_cli': 'ipc:///tmp/snp-svr-1.ipc',
            'scheduler_url': args.server_url,

            'trainer_cls': tnoppt.TrainerOPPytorch,
            'trainer_cfg': {
                'kn': args.world_size,
                'ti': args.rank,
                'ctx': ctx,
                'net': net,
                'net_init': None,
                'opt': opt,
                #'loss': F.nll_loss,  # use the original loss func
                'loss': WrpDoLoss(net, ctx, F.nll_loss),  # or use the loss wrapper
                'mtrc': None,
                'optimizer_params': None,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'dataloader_creator': get_train_dataloader,
                'dataloader_creator_args': (),
                'dataloader_creator_kwargs': {'kn': args.world_size,
                                              'rank': args.rank,
                                              'batch_size': args.batch_size,
                                              'shuffle': True,
                                              'args': args},
                'use_dist_data_sampler': False,
                'log_interval': args.log_interval,
            },
        }

        svc = tndsnc.SVCWorker(cfg)

        hook_do_test = HookDoTest(args, net, ctx)
        svc.svr.trnr.register_hook_after_epoch(hook_do_test)

        svc.loop()

    else:
        print('wrong role: %s' % (role))


# python train_mnist_mpnn_21_one_grp.py -r scheduler -w 2 -k 0 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc -b 1000
# python train_mnist_mpnn_21_one_grp.py -r worker -w 2 -k 0 --gpus 0 -u ipc:///tmp/ngn-wkr-0.ipc -s ipc:///tmp/snp-svr-1.ipc -b 1000
# python train_mnist_mpnn_21_one_grp.py -r worker -w 2 -k 1 --gpus 1 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc -b 1000
