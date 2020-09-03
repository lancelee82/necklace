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

The code is [here](https://github.com/lancelee82/necklace/examples/mnist/pytorch/).

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


