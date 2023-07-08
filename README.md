# necklace

```
\-o-O-0-O-o-/
```

Distributed deep learning framework based on pytorch/mxnet/numba and nccl.


## Arch

```
|  SCHD  ||  WKR  |  WKR  |  WKR  |
-----------------------------------
|     Distributed DL Trainer      |
===================================
|          RPC Framework          |
-----------------------------------
|         ZeroMQ  +  NCCL         |
-----------------------------------
|  GPU |  GPU |  GPU |  GPU | GPU |
```

There are two types of processes in necklace for now:
* the scheduler, manages the whole process of the training
* some workers, really do the training on hardware with a specific deep learning framework

They communicate with each other by two types of messages:
* the control messages are through a RPC mechanism based on ZeroMQ
* the training informations such as gradients and weights of model are through NCCL


## Install

### Requirements

* CUDA + NCCL

Please reference Nvidia documents.

* pynccl

The pynccl repo is [here](https://github.com/lancelee82/pynccl), or just use pip

```
pip install pynccl
```

* zerorpc

Necklace implements a RPC framework based on [ZeroRPC](http://www.zerorpc.io/) 
called [OneRPC](https://github.com/lancelee82/necklace/tree/master/necklace/rpc).


## Examples

### DP (Data-Parallel)

#### mnist + pytorch

The hello-world example is also the mnist project and PyTorch framework based,
The code is [here](https://github.com/lancelee82/necklace/tree/master/examples/dp/mnist/pytorch).

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


### MP (Model-Parallel)

#### mnist + pytorch

The example is also the mnist project and PyTorch framework based,
The code is [here](https://github.com/lancelee82/necklace/tree/master/examples/mp/mnist/pytorch).

On a server with some GPUs, run scheduler

```
$ python train_mnist_mp_5_nklc.py -r scheduler -w 2 -k 0 --epochs 3 -u ipc:///tmp/snp-svr-1.ipc -b 256
```

and then run two workers

```
$ python train_mnist_mp_5_nklc.py -r worker -w 2 -k 0 --gpus 0 -u ipc:///tmp/ngn-wkr-0.ipc -s ipc:///tmp/snp-svr-1.ipc -b 256
```

```
$ python train_mnist_mp_5_nklc.py -r worker -w 2 -k 1 --gpus 2 -u ipc:///tmp/ngn-wkr-1.ipc -s ipc:///tmp/snp-svr-1.ipc -b 256
```


## TODO

- [x] DP (Data Parallelism)
- [x] MP (Model Parallelism)
- [x] PP (Pipeline Parallelism)
- [ ] ZeRO ([ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf))

