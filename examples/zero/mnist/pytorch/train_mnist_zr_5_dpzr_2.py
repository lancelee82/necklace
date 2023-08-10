""""""
# ===========================================================================
# ===========================================================================

from __future__ import print_function
import os
import sys
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# ----------------------------------------------------------------------------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURR_DIR))))  # ../../../../
sys.path.insert(0, ROOT_DIR)

import necklace
from necklace.trainer import tndszd
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
parser.add_argument('--batch-size', '-b', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')

attach_nk_args_parser(parser)

args = parser.parse_args()


# ----------------------------------------------------------------------------
# init nccl groups (dp + zr)
# TODO: put this after the get_dataloader to avoid calling in subprocess

from necklace.cuda import ncclgrp
pg_cfg = {
    'world_size': args.world_size,
    'my_rank': args.rank,
    'dp_size': args.dp_size,
    #'mp_size': args.mp_size,
    #'pp_size': args.pp_size,
    'zr_size': args.zr_size,
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

class NetPart1(nn.Module):
    def __init__(self):
        super(NetPart1, self).__init__()
        #d1 = 32
        d1 = 768  # big model :p
        self.conv1 = nn.Conv2d(1, d1, 3, 1)
        self.conv2 = nn.Conv2d(d1, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

        #self.dropout2 = nn.Dropout(0.2)
        ##d6 = 128
        #d6 = 5120 #10240  # big model :p
        #self.fc1 = nn.Linear(9216, d6)
        #self.fc2 = nn.Linear(d6, 9216)

    def forward(self, x):
        # x : 1x28x28
        x = self.conv1(x)  # d1x26x26
        x = F.relu(x)
        x = self.conv2(x)  # 64x24x24
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 64x12x12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # 9216

        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.dropout2(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        return x


class NetPart2_1(nn.Module):
    def __init__(self):
        super(NetPart2_1, self).__init__()
        self.dropout2 = nn.Dropout(0.2)
        #d6 = 128
        d6 = 10240  # big model :p
        self.fc1 = nn.Linear(9216, d6)
        self.fc2 = nn.Linear(d6, 9216)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class NetPart2_2(nn.Module):
    def __init__(self):
        super(NetPart2_2, self).__init__()
        self.dropout2 = nn.Dropout(0.1)
        #d6 = 128
        d6 = 10240  # big model :p
        self.fc1 = nn.Linear(9216, d6)
        self.fc2 = nn.Linear(d6, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.submdl_1 = NetPart1()
        #self.submdl_2_1 = NetPart2_1()
        self.submdl_2_2 = NetPart2_2()


# TODO: automatically built with other infos ==> mputils.MpTrnrModulesMap
def get_net_map():
    net_map = OrderedDict()

    net = Net()

    net_map[0] = {'net': net.submdl_1, 'net_map_order': 0, 'worker_rank': 0}
    #net_map[1] = {'net': [net.submdl_2,], 'net_map_order': 1, 'worker_rank': 1}
    #net_map[1] = {'net': net.submdl_2_1, 'net_map_order': 1, 'worker_rank': 1}
    net_map[1] = {'net': net.submdl_2_2, 'net_map_order': 1, 'worker_rank': 1}

    return net_map


# ----------------------------------------------------------------------------

def get_data_loader(*args, **kwargs):
    batch_size = kwargs.get('batch_size', 64)

    # NOTE: for distributed sampler/dataloader
    kn = kwargs.get('kn', 1)
    rank = kwargs.get('rank', 0)  # rank
    shuffle = kwargs.get('shuffle', True)

    pg_mp = ncclgrp.get_nccl_group_mp()
    pg_dp = ncclgrp.get_nccl_group_dp()

    mp_world_size = ncclgrp.get_world_size(pg_mp)
    mp_rank = ncclgrp.get_rank(pg_mp)
    dp_world_size = ncclgrp.get_world_size(pg_dp)
    dp_rank = ncclgrp.get_rank(pg_dp)

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    #kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    # ============================================================
    # NOTE: here we use DistributedSampler with necklace
    # ============================================================
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=kn, rank=rank, shuffle=shuffle)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'sampler': train_sampler}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, **kwargs)

    # TODO: [==============================================================================]
    """
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dp_world_size, rank=dp_rank, shuffle=shuffle)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'sampler': train_sampler, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, **kwargs)
    """

    test_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': shuffle}
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
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
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


class HookAfterOneOptStep(hookbase.NklcHookBase):
    def __init__(self, net, ctx, *args, **kwargs):
        super(HookAfterOneOptStep, self).__init__(net, ctx, *args, **kwargs)

    def __call__(self, trnr, *args, **kwargs):
        print('-' * 40, trnr.e, '--', trnr.i)


class WrpDoLoss(wrpbase.NklcWrapperBase):
    def __init__(self, net, ctx, loss_fn, *args, **kwargs):
        super(WrpDoLoss, self).__init__(net, ctx, *args, **kwargs)
        self.loss_fn = loss_fn

    def __call__(self, output, target, *args, **kwargs):
        loss = self.loss_fn(output, target)
        print('loss', loss.cpu().item())
        return loss


# ----------------------------------------------------------------------------

from mlp_mixer import MLPMixer, MLPMixerInp, MLPMixerBody, MLPMixerHead

class MLPMxr(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super(MLPMxr, self).__init__()

        self.submdl_1 = MLPMixerInp(in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim)
        self.submdl_2 = MLPMixerBody(in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim)
        #self.submdl_2_2 = MLPMixerBody(in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim)
        self.submdl_3 = MLPMixerHead(in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim)

    def forward(self, x):
        x = self.submdl_1(x)
        x = self.submdl_2(x)
        #x = self.submdl_2_2(x)
        x = self.submdl_3(x)
        return x


# TODO: automatically built with other infos ==> mputils.MpTrnrModulesMap
def get_net_map____():  # TODO: sub-num: 3---->2  [=====================================]

    #model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
    #                 dim=512, depth=80, token_dim=256, channel_dim=2048)
    #model = MLPMixer(in_channels=3, image_size=28, patch_size=4, num_classes=10,
    #                 dim=512, depth=80, token_dim=256, channel_dim=2048)

    #net = MLPMxr(in_channels=1, image_size=28, patch_size=4, num_classes=10,
    #                 dim=512, depth=30, token_dim=256, channel_dim=2048)  # NOTE: for wkrs:3

    net = MLPMxr(in_channels=1, image_size=28, patch_size=4, num_classes=10,
                 #dim=1024, depth=20, token_dim=512, channel_dim=4096)
                 dim=256, depth=10, token_dim=128, channel_dim=1024)
                 #dim=64, depth=6, token_dim=32, channel_dim=256)  # for small model

    net_map = OrderedDict()

    net_map[0] = {'net': net.submdl_1, 'net_map_order': 0, 'worker_rank': 0}
    net_map[1] = {'net': net.submdl_2, 'net_map_order': 1, 'worker_rank': 1}
    #net_map[2] = {'net': net.submdl_2_2, 'net_map_order': 2, 'worker_rank': 2}
    net_map[2] = {'net': net.submdl_3, 'net_map_order': 3, 'worker_rank': 3}

    return net_map


# ----------------------------------------------------------------------------

def get_net_ctx(args, net_map):
    gpu_dev_i = int(args.gpu_rank)
    ctx = torch.device(gpu_dev_i)

    #zr_rank = args.rank
    zr_rank = nccl_groups_cfg_map.get('zr_group', {}).get('rank')  # TODO: to global
    print('zr_rank: ', zr_rank)

    model = net_map[zr_rank]['net']
    #optimizer = optim.SGD(model.parameters(),
    #                      lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # TODO: when model is a list of sub-modules

    return model, optimizer, ctx


def get_opt(model, *aargs, **kwargs):
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum)
    return optimizer


# ----------------------------------------------------------------------------
from necklace.frmwrk.pytorch.ptutils import prnt_md

def print_net_map(netmap):
    for k, m in netmap.items():
        net = m['net']
        print('-=' * 20, m['net_map_order'])
        prnt_md(net)


if __name__ == '__main__':

    ilog.il_add(tndszd.ILOG_I_ZR_SVR_WKR)
    #ilog.il_add(tndszd.ILOG_I_ZR_SVC_WKR)
    ilog.il_add(tndszd.ILOG_I_ZR_SVR_SCH)
    #ilog.il_add(tndszd.ILOG_I_ZR_SVC_SCH)

    net_map = get_net_map()
    print_net_map(net_map)

    role = args.role

    if role == 'scheduler':
        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'wdsz': args.world_size,
            'rank': args.rank,

            'epochs': args.epochs,

            'net_map': net_map,

            # ========================================================
            # NOTE: DP + ZR
            # ========================================================
            'nccl_allreduce_typ': 'dpzr', #'grad',  # 'weig'  # TODO: [====================================]

            'nccl_groups_cfg_map': nccl_groups_cfg_map,  # only my
            'nccl_groups_grp_map': nccl_groups_grp_map,  # all grps
            'nccl_group_main_grp_list': nccl_group_main_grp_list,

        }

        svc = tndszd.MPSVCScheduler(cfg)
        svc.loop()

    elif role == 'worker':

        gpu_dev_i = int(args.gpu_rank)

        # ========================================================
        # do necklace init (mp/cuda) here only in worker
        # ========================================================
        necklace.init(gpu_dev_i)

        # set cuda device of pytorch
        torch.cuda.set_device(gpu_dev_i)

        from necklace.trainer import zroppd

        net, opt, ctx = get_net_ctx(args, net_map)

        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'wdsz': args.world_size,
            'rank': args.rank,

            #'svr_cli': 'ipc:///tmp/snp-svr-1.ipc',
            'scheduler_url': args.server_url,

            'trainer_cls': zroppd.TrainerMPOPPytorch,
            'trainer_cfg': {
                'kn': args.world_size,
                'ti': args.rank,
                'ctx': ctx,
                'net': net,#net_map[args.rank],
                'net_map': net_map,  # NOTE: now should on CPU
                'net_init': None,
                #'net_map_order': args.rank,  # TODO: [==============================]
                'net_map_order': nccl_groups_cfg_map.get('zr_group', {}).get('rank'),
                'net_map_maxlen': len(net_map.keys()),
                'opt': opt,
                'optimizer_creator': get_opt,
                'optimizer_creator_args': (),
                'optimizer_creator_kwargs': {},
                #'loss': F.nll_loss,  # use the original loss func
                #'loss': WrpDoLoss(net, ctx, F.nll_loss),  # or use the loss wrapper
                'loss': WrpDoLoss(net, ctx, nn.CrossEntropyLoss()),  # NOTE: model without log_softmax
                'mtrc': None,
                'optimizer_params': None,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'dataloader_creator': get_train_dataloader,
                'dataloader_creator_args': (),
                'dataloader_creator_kwargs': {'kn': args.world_size,
                                              'rank': args.rank,  # TODO: [==============================]
                                              'batch_size': args.batch_size,
                                              'shuffle': False,  # NOTE: for DistributedSampler on diff node
                                              'args': args},
                'use_dist_data_sampler': False,
                'log_interval': args.log_interval,
            },
        }

        svc = tndszd.MPSVCWorker(cfg)

        # TODO: for now we can not run test because the model is a part,
        #       and we must collect the whole model to a place to run it
        #hook_do_test = HookDoTest(args, net, ctx)
        #svc.svr.trnr.register_hook_after_epoch(hook_do_test)

        hook_after_a_step = HookAfterOneOptStep(net, ctx)
        svc.svr.trnr.register_hook_after_do_a_train_step(hook_after_a_step)

        svc.loop()

    else:
        print('wrong role: %s' % (role))


"""
python train_mnist_zr_5_dpzr_2.py -r scheduler -w 4 -k 0 -t "dp+zr" -dpsz 2 -zrsz 2 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc -b 300
python train_mnist_zr_5_dpzr_2.py -r worker -w 4 -k 0 -g 0 -t "dp+zr" -dpsz 2 -zrsz 2 -u ipc:///tmp/ngn-wkr-0.ipc -s ipc:///tmp/snp-svr-1.ipc -b 300
python train_mnist_zr_5_dpzr_2.py -r worker -w 4 -k 1 -g 1 -t "dp+zr" -dpsz 2 -zrsz 2 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc -b 300
python train_mnist_zr_5_dpzr_2.py -r worker -w 4 -k 2 -g 2 -t "dp+zr" -dpsz 2 -zrsz 2 -u ipc:///tmp/ngn-wkr-2.ipc -s ipc:///tmp/snp-svr-1.ipc -b 300
python train_mnist_zr_5_dpzr_2.py -r worker -w 4 -k 3 -g 3 -t "dp+zr" -dpsz 2 -zrsz 2 -u ipc:///tmp/ngn-wkr-3.ipc -s ipc:///tmp/snp-svr-1.ipc -b 300
"""
