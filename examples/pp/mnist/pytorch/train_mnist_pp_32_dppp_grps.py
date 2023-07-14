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
from necklace.trainer import tndspp
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


# ----------------------------------------------------------------------------
# init nccl groups (dp + mp)
# TODO: put this after the get_dataloader to avoid calling in subprocess

from necklace.cuda import ncclgrp
#ncclgrp.set_nccl_group_main_world_size(args.world_size)
#ncclgrp.set_nccl_group_main_rank(args.rank)
pg_cfg = {
    'world_size': args.world_size,
    'my_rank': args.rank,
    'dp_size': args.dp_size,
    #'mp_size': args.mp_size,
    'pp_size': args.pp_size,
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

class Net_Ori(nn.Module):
    def __init__(self):
        super(Net_Ori, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward_ori(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def md_split_to_devices(self, devs=[]):
        if not devs:
            raise Exception('no multi devices')

        self.devs = devs
        device1, device2 = devs[0], devs[1]  # TODO:

        self.conv1 = self.conv1.to(device1)
        self.conv2 = self.conv2.to(device1)
        self.dropout1 = self.dropout1.to(device1)

        self.fc1 = self.fc1.to(device2)
        self.dropout2 = self.dropout2.to(device2)
        self.fc2 = self.fc2.to(device2)

    def forward(self, x):
        device1, device2 = self.devs[0], self.devs[1]  # TODO:

        x = x.to(device1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        # ----------------------------------------
        x = x.to(device2)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetPart1(nn.Module):
    def __init__(self):
        super(NetPart1, self).__init__()
        #d1 = 32
        d1 = 768  # big model :p
        self.conv1 = nn.Conv2d(1, d1, 3, 1)
        self.conv2 = nn.Conv2d(d1, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x


class NetPart2(nn.Module):
    def __init__(self):
        super(NetPart2, self).__init__()
        self.dropout2 = nn.Dropout2d(0.5)
        #d6 = 128
        d6 = 10240  # big model :p
        self.fc1 = nn.Linear(9216, d6)
        self.fc2 = nn.Linear(d6, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.submdl_1 = NetPart1()
        self.submdl_2 = NetPart2()


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

    # ============================================================
    # NOTE: for MPv (MP with vertical split) without DP, only the
    #       start worker (rank-0) needs the dataloader, and here
    #       for simplified we create dataloaders in all workers
    # ============================================================
    """
    kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, **kwargs)
    """

    # TODO: [==============================================================================]
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dp_world_size, rank=dp_rank, shuffle=shuffle)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'sampler': train_sampler, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, **kwargs)

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

# TODO: automatically built with other infos ==> mputils.MpTrnrModulesMap
def get_net_map():
    net_map = OrderedDict()

    net = Net()

    net_map[0] = {'net': net.submdl_1, 'net_map_order': 0, 'worker_rank': 0}
    #net_map[1] = {'net': [net.submdl_2,], 'net_map_order': 1, 'worker_rank': 1}
    net_map[1] = {'net': net.submdl_2, 'net_map_order': 1, 'worker_rank': 1}

    return net_map


def get_net_ctx(args, net_map):
    gpu_dev_i = int(args.gpu_rank)
    ctx = torch.device(gpu_dev_i)

    #pp_rank = args.rank
    pp_rank = nccl_groups_cfg_map.get('pp_group', {}).get('rank')  # TODO: to global
    print('pp_rank: ', pp_rank)

    model = net_map[pp_rank]['net']
    #optimizer = optim.SGD(model.parameters(),
    #                      lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # TODO: when model is a list of sub-modules

    return model, optimizer, ctx


if __name__ == '__main__':

    #ilog.il_add(tndspp.ILOG_I_MP_SVR_WKR)
    #ilog.il_add(tndspp.ILOG_I_MP_SVC_WKR)
    #ilog.il_add(tndspp.ILOG_I_MP_SVR_WKR_PRT)
    #ilog.il_add(tndspp.ILOG_I_MP_SVC_WKR_PRT)
    ilog.il_add(tndspp.ILOG_I_MP_SVR_SCH)
    #ilog.il_add(tndspp.ILOG_I_MP_SVC_SCH)
    #ilog.il_add(tndspp.ILOG_I_MP_SVR_SCH_PRT)
    #ilog.il_add(tndspp.ILOG_I_MP_SVC_SCH_PRT)

    net_map = get_net_map()

    role = args.role

    if role == 'scheduler':
        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'wdsz': args.world_size,
            'rank': args.rank,

            'epochs': args.epochs,
            'nccl_allreduce_typ': 'grad',  # 'weig'

            'net_map': net_map,

            'nccl_groups_cfg_map': nccl_groups_cfg_map,  # only my
            'nccl_groups_grp_map': nccl_groups_grp_map,  # all grps
            'nccl_group_main_grp_list': nccl_group_main_grp_list,
        }

        svc = tndspp.MPSVCScheduler(cfg)
        svc.loop()

    elif role == 'worker':

        gpu_dev_i = int(args.gpu_rank)

        # ========================================================
        # do necklace init (mp/cuda) here only in worker
        # ========================================================
        necklace.init(gpu_dev_i)

        # set cuda device of pytorch
        torch.cuda.set_device(gpu_dev_i)

        from necklace.trainer import ppoppt

        net, opt, ctx = get_net_ctx(args, net_map)

        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'wdsz': args.world_size,
            'rank': args.rank,

            #'svr_cli': 'ipc:///tmp/snp-svr-1.ipc',
            'scheduler_url': args.server_url,

            'trainer_cls': ppoppt.TrainerMPOPPytorch,
            'trainer_cfg': {
                'kn': args.world_size,
                'ti': args.rank,
                'ctx': ctx,
                'net': net,#net_map[args.rank],
                'net_init': None,
                #'net_map_order': args.rank,  # TODO: [==============================]
                'net_map_order': nccl_groups_cfg_map.get('pp_group', {}).get('rank'),
                'net_map_maxlen': len(net_map.keys()),
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
                                              'rank': args.rank,  # TODO: [==============================]
                                              'batch_size': args.batch_size,
                                              'shuffle': True,
                                              'args': args},
                'use_dist_data_sampler': False,
                'log_interval': args.log_interval,
            },
        }

        svc = tndspp.MPSVCWorker(cfg)

        # TODO: for now we can not run test because the model is a part,
        #       and we must collect the whole model to a place to run it
        #hook_do_test = HookDoTest(args, net, ctx)
        #svc.svr.trnr.register_hook_after_epoch(hook_do_test)

        hook_after_a_step = HookAfterOneOptStep(net, ctx)
        svc.svr.trnr.register_hook_after_do_a_train_step(hook_after_a_step)

        svc.loop()

    else:
        print('wrong role: %s' % (role))


# python train_mnist_mp_5_nklc.py -r scheduler -w 2 -k 0 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc -b 256
# python train_mnist_mp_5_nklc.py -r worker -w 2 -k 0 -g 0 -u ipc:///tmp/ngn-wkr-0.ipc -s ipc:///tmp/snp-svr-1.ipc -b 256
# python train_mnist_mp_5_nklc.py -r worker -w 2 -k 1 -g 1 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc -b 256

"""

python train_mnist_pp_32_dppp_grps.py -r scheduler -w 4 -k 0 -t "dp+pp" -dpsz 2 -ppsz 2 --epochs 3 -u tcp://192.168.58.193:11001

python train_mnist_pp_32_dppp_grps.py -r worker -w 4 -k 0 -g 0 -t "dp+pp" -dpsz 2 -ppsz 2 -u tcp://192.168.58.193:12000 -s tcp://192.168.58.193:11001

python train_mnist_pp_32_dppp_grps.py -r worker -w 4 -k 1 -g 1 -t "dp+pp" -dpsz 2 -ppsz 2 -u tcp://192.168.58.193:12001 -s tcp://192.168.58.193:11001

python train_mnist_pp_32_dppp_grps.py -r worker -w 4 -k 2 -g 2 -t "dp+pp" -dpsz 2 -ppsz 2 -u tcp://192.168.58.193:12002 -s tcp://192.168.58.193:11001

python train_mnist_pp_32_dppp_grps.py -r worker -w 4 -k 3 -g 0 -t "dp+pp" -dpsz 2 -ppsz 2 -u tcp://192.168.58.192:12001 -s tcp://192.168.58.193:11001

"""

