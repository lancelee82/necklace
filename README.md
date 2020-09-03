# necklace

```
\-o-O-0-O-o-/
```

Distributed deep learning framework based on mxnet/pytorch/numba and nccl.


## Arch

```
|  SCHD  ||  WKR  |  WKR  |  WKR  |
-----------------------------------
|     Distributed DL Trainer      |
===================================
|           RPC Framework         |
-----------------------------------
|          ZeroMQ  +  NCCL        |
-----------------------------------
|  GPU |  GPU |  GPU |  GPU | GPU |
```

There are two types of processes in necklace for now,
the scheduler, manages the whole process of the training,
and some workers, really do the training on hardware with a specific deep learning framework.
They communicate with each other by two types of messages,
the control messages are through a RPC mechanism based on ZeroMQ,
and the training data exchange such as gradients and weights of model are through NCCL.


## Install

### Requirements

* CUDA + NCCL

Please reference Nvidia documents.

* pynccl

The pynccl repo is [here](https://github.com/lancelee82/pynccl), or just use pip

```
pip install pynccl
```


## Examples

### mnist + pytorch

The helloworld example is the mnist dataset and PyTorch framework based,
The code is [here](https://github.com/lancelee82/necklace/tree/master/examples/mnist/pytorch).

On a server with some GPUs, run scheduler

```
$ python train_ngn_mnist_1.py -r scheduler -w 2 -k 0 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc
```

and then run two workers

```
$ python train_ngn_mnist_1.py -r worker -w 2 -k 0 --gpus 0 -u ipc:///tmp/ngn-wkr-0.ipc -s ipc:///tmp/snp-svr-1.ipc
```

```
$ python train_ngn_mnist_1.py -r worker -w 2 -k 1 --gpus 1 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc
```


## TODO


