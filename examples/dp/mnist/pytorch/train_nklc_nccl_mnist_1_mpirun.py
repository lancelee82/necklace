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
#necklace.init(gpu_dev_i)
from necklace.trainer import tndsnc
from necklace.rpc import ilog
from necklace.utils import hookbase
from necklace.utils import wrpbase
from necklace.utils import argutils


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
parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--gpus', '-g', dest='gpu_ids', help='GPU device to train with',
                    default='0,1,2,3', type=str)

parser.add_argument('--url', '-u', type=str, default='ipc:///tmp/ngn-wkr-1.ipc',
                    help='worker url')
parser.add_argument('--server-url', '-s', type=str, default='ipc:///tmp/ngn-svr-1.ipc',
                    help='server url')
parser.add_argument('--role', '-r', type=str, default='worker',
                    help='node role')
parser.add_argument('--world-size', '-w', type=int, default=1,
                    help='workers number (default is 1)')
parser.add_argument('--rank', '-k', type=int, default=0,
                    help='worker rank (required)')
parser.add_argument('--nslot', '-n', type=int, default=1,
                    help='slot num per host (required)')

parser.add_argument('--rpc_ifname', '-ri', type=str, default='',
                    help='rpc ifname')
parser.add_argument('--rpc_ip_prefix', '-rp', type=str, default='',
                    help='rpc ip prefix : 192.168.11.')
parser.add_argument('--comm_ifname', '-ci', type=str, default='',
                    help='comm ifname')

parser.add_argument('--comm_type', '-x', type=str, default='mpi',
                    help='communication type [mpi,nccl]')

args = parser.parse_args()

#gpu_dev_i = int(args.gpu_ids.split(',')[0])  # TODO: not used for now when using MPI !?

# ======================================================
# NOTE: for nccl ENV when using mpirun
# TODO: or, to:
#        args = argutils.args_remake(args)
# ======================================================
if args.comm_type == 'nccl':
    if args.comm_ifname and not os.environ.get('NCCL_SOCKET_IFNAME'):
        os.environ['NCCL_SOCKET_IFNAME'] = args.comm_ifname


# ----------------------------------------------------------------------------
# original training code

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 10000, kernel_size=5)  # NOTE: larger model
        self.conv2 = nn.Conv2d(10000, 20, kernel_size=5)
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


def get_net_ctx(args, gpu_dev_i):
    #gpu_dev_i = int(args.gpu_ids.split(',')[0])
    ctx = torch.device(gpu_dev_i)

    model = Net().to(ctx)
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
            'nccl_allreduce_typ': 'grad',  # 'weig'
        }

        svc = tndsnc.SVCScheduler(cfg)
        svc.loop()

    elif role == 'worker':

        # ========================================================
        # NOTE: only import MPI in workers
        # ========================================================
        from necklace.pympi import api as mpiwrp
        nc = mpiwrp.MpiWrp()

        # refresh the args
        args.world_size = nc.world_size
        args.rank = nc.rank

        # NOTE: the zrpc url should be diff by rank [!!!]
        # TODO: or use @{RANK} in args and replace by rank here [===========]
        #args.url = args.url + str(args.rank)

        # TODO: how to get the gpu_dev_i on diff hosts ??? [=================]
        # <1> here we assumpt every host has the same number of nslot (gpus),
        # so we can get gpu_dev_i from 0 to nslot-1
        # <2> or, we need hosts topology by setting like -H host1:n1,host2:n2,
        # and then get my hostname then my nslot, and my rank ???
        #gpu_dev_i = args.rank % args.nslot

        # ========================================================
        # remake the args (with the newest infos)
        # ========================================================
        args = argutils.args_remake(args)

        print('wkr_url:', args.url)

        gpu_dev_i = args.gpu_dev_i

        # ========================================================
        # do necklace init (mp/cuda) here only in worker
        # ========================================================
        necklace.init(gpu_dev_i)

        # set cuda device of pytorch
        torch.cuda.set_device(gpu_dev_i)

        from necklace.trainer import tnoppt

        net, opt, ctx = get_net_ctx(args, gpu_dev_i)

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
                'nc': nc,  # NOTE: the nc is created now, so we pass it to the trainer
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
                                              'shuffle': False,  # NOTE: for DistributedSampler on diff node
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


#-----------------------------------------------------------------------------------------------------
# mpi on one same node (venv)
# python train_nklc_nccl_mnist_1_mpirun.py -r scheduler -w 2 -k 0 -n 2 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc
# python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 2 -k 0 --gpus 0 -x nccl -ci eno33 -n 2 -u ipc:///tmp/ngn-wkr-0.ipc -s ipc:///tmp/snp-svr-1.ipc
# python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 2 -k 1 --gpus 1 -x nccl -ci eno33 -n 2 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc

# mpirun -np 2 /home/user/data0/proj/py/venv/vpt113g/bin/python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 2 -k 0 --gpus 1 -n 2 -x nccl -ci eno33 -u ipc:///tmp/ngn-wkr-@{LOCAL_RANK}.ipc -s ipc:///tmp/snp-svr-1.ipc


#-----------------------------------------------------------------------------------------------------
# mpi on two nodes (venv)
# python train_nklc_nccl_mnist_1_mpirun.py -r scheduler -w 2 -k 0 -n 2 --epochs 3 -u tcp://192.168.58.193:11001
# python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 2 -k 0 --gpus 0 -n 2 -x nccl -ci eno33 -u tcp://192.168.58.193:12000 -s tcp://192.168.58.193:11001
# python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 2 -k 1 --gpus 1 -n 2 -x nccl -ci eno33 -u tcp://192.168.58.193:12001 -s tcp://192.168.58.193:11001

# mpirun -np 2 -H a100svr003:1,worker00:1 /home/user/data0/proj/py/venv/vpt113g/bin/python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 2 -k 0 --gpus 0 -n 1 -x nccl -ci eno33 -ri eno33 -u tcp://@{MY_IP}:1200@{LOCAL_RANK} -s tcp://192.168.58.193:11001

# --------------------------------------------------------
# python train_nklc_nccl_mnist_1_mpirun.py -r scheduler -w 4 -k 0 -n 2 --epochs 3 -u tcp://192.168.58.193:11001

# mpirun -np 4 -H a100svr003:2,worker00:2 /home/user/data0/proj/py/venv/vpt113g/bin/python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 4 -k 0 --gpus 0 -n 2 -x nccl -ci eno33 -ri eno33 -u tcp://@{MY_IP}:1200@{LOCAL_RANK} -s tcp://192.168.58.193:11001

# --------------------------------------------------------
# python train_nklc_nccl_mnist_1_mpirun.py -r scheduler -w 6 -k 0 -n 3 --epochs 3 -u tcp://192.168.58.193:11001

# mpirun -np 6 -H a100svr003:3,worker00:3 /home/user/data0/proj/py/venv/vpt113g/bin/python train_nklc_nccl_mnist_1_mpirun.py -r worker -w 6 -k 0 --gpus 0 -n 3 -x nccl -ci eno33 -ri eno33 -u tcp://@{MY_IP}:1200@{LOCAL_RANK} -s tcp://192.168.58.193:11001

