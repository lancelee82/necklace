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

parser.add_argument('--url', '-u', type=str, default='ipc:///tmp/ngn-svr-1.ipc',
                    help='server url')
parser.add_argument('--server-url', '-s', type=str, default='ipc:///tmp/ngn-svr-1.ipc',
                    help='server url')
parser.add_argument('--role', '-r', type=str, default='worker',
                    help='node role')
parser.add_argument('--world-size', '-w', type=int, default=1,
                    help='workers number (default is 1)')
parser.add_argument('--rank', '-k', type=int, default=0,
                    help='worker rank (required)')

args = parser.parse_args()


# ----------------------------------------------------------------------------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURR_DIR))))  # ../../../../
sys.path.insert(0, ROOT_DIR)

from necklace.data import sampler as _sampler
#from necklace.data import dtmptp as dtldp
#from necklace.data import dtmptv as dtldp
#from necklace.data import dtqptp as dtldp
from necklace.data import dtqpto as dtldp
from necklace.data import batchify
from necklace.trainer import tndsnc


# ----------------------------------------------------------------------------
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


def get_net_ctx(args):
    gpu_dev_i = int(args.gpu_ids.split(',')[0])
    ctx = torch.device(gpu_dev_i)

    model = Net().to(ctx)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum)

    return model, optimizer, ctx

# data

def get_data_loader(*aargs, **kwargs):
    args = kwargs.get('args', {})
    batch_size = kwargs.get('batch_size', 64)

    train_ds = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_loader = dtldp.DataLoaderSharedMemory(
        train_ds,
        batch_size=batch_size,
        data_shape=(1, 28, 28),
        shuffle=False,#True,  # no need for dist dataloader
        last_batch='discard',
        batchify_fn=batchify.flt2int_bchfn,
        inter_batchify_fn=batchify.pt_batchify_fn,
        part_num=20,
        num_workers=4)
        #num_workers=1)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def get_train_data(*args, **kwargs):
    train_loader, test_loader = get_data_loader(*args, **kwargs)
    return train_loader


# ----------------------------------------------------------------------------
# TODO: add to trainer
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
from necklace.rpc import ilog

#ilog.il_add(500001)
#ilog.il_add(500002)


if __name__ == '__main__':

    # -------------------------------------------------------------------
    import multiprocessing as mp
    print(mp.get_start_method())
    #mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    print(mp.get_start_method())
    # -------------------------------------------------------------------

    role = args.role

    if role == 'scheduler':
        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
            'rank': args.rank,

            'epochs': args.epochs,
            'nccl_allreduce_typ': 'grad',  # 'weig'
        }

        svc = tndsnc.SVCScheduler(cfg)
        svc.loop()

    elif role == 'worker':
        # NOTE: here we should do selecting gpu device firstly before
        # any other import and use of numba, otherwise the error of
        # core dump maybe occur sometimes
        gpu_dev_i = int(args.gpu_ids.split(',')[0])
        from necklace.cuda import nbutils
        nbutils.cuda_select_device(gpu_dev_i)

        # set cuda device of pytorch
        torch.cuda.set_device(gpu_dev_i)

        from necklace.trainer import tnoppt

        net, opt, ctx = get_net_ctx(args)

        cfg = {
            'role': role,
            'url': args.url,
            'workers': args.world_size,
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
                'loss': F.nll_loss,
                'mtrc': None,
                'optimizer_params': None,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'dataloader_creator': get_train_data,
                'dataloader_creator_args': (),
                'dataloader_creator_kwargs': {'batch_size': args.batch_size,
                                              'kn': args.world_size,
                                              'rank': args.rank,
                                              'shuffle': True,
                                              'args': args},
                'use_dist_data_sampler': False,#True,  # TODO: reset_distdt_indices
                'log_interval': args.log_interval,
                #'grad_comp_cfg': {'grad_size_threshold': args.batch_size * 100000,
                #                  'grad_comp_threshold': 0.5},
            },
        }

        svc = tndsnc.SVCWorker(cfg)
        svc.loop()

        # TODO:
        train_loader, test_loader = get_data_loader(batch_size=args.batch_size)
        test(args, net, ctx, test_loader)

    else:
        print('wrong role: %s' % (role))


# python train_ngn_mnist_1.py -r scheduler -w 2 -k 0 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc
# python train_ngn_mnist_1.py -r worker -w 2 -k 0 --gpus 0 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc

# python train_ngn_mnist_1.py -r scheduler -w 8 -k 0 -u tcp://10.60.242.136:9399
# ./train_workers.sh 4
# ./train_workers.sh 4
#  python train_ngn_mnist_1.py -r worker -w 2 -k 0 -u tcp://10.60.242.136:9300 -s tcp://10.60.242.136:9399 --gpus 0 --lr 0.05
