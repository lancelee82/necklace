# necklace

```
\-o-O-0-O-o-/
```

Distributed deep learning framework based on pytorch/numba/nccl and zeromq.


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


## Trainer Mode

```
TRN_MODE_L = [
    'no'
    'dp',
    'pp',
    'mp',
    'zr',
    'dp+pp',
    'dp+mp',
    'pp+mp',  # x
    'dp+pp+mp',
    'dp+zr',
    'zr+pp',  # x
    'zr+mp',
    'zr+pp+mp',  # x
    'dp+zr+mp',
    'dp+zr+pp',  # x
    'dp+zr+pp+mp',  # x
]
```


## Examples

All the examples are [here](https://github.com/lancelee82/necklace/tree/master/examples/).

Note that all the run commands are at the end of the files.

For one example, with trainer mode "dp+pp+mp": [code](https://github.com/lancelee82/necklace/tree/master/examples/pp/mnist/pytorch/train_mnist_pp_81_dpppmp_3.py)

On a server, run scheduler

```
$ python train_mnist_pp_81_dpppmp_3.py -r scheduler -w 8 -k 0 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 --epochs 3 -u tcp://192.168.58.193:11001 -b 100
```

On some servers with some GPUs, run some workers

```
$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 0 -g 0 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.193:12000 -s tcp://192.168.58.193:11001 -b 100
$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 1 -g 1 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.193:12001 -s tcp://192.168.58.193:11001 -b 100
$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 2 -g 2 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.193:12002 -s tcp://192.168.58.193:11001 -b 100

$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 3 -g 0 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.192:12000 -s tcp://192.168.58.193:11001 -b 100
$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 4 -g 1 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.192:12001 -s tcp://192.168.58.193:11001 -b 100
$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 5 -g 2 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.192:12002 -s tcp://192.168.58.193:11001 -b 100

$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 6 -g 0 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.194:12000 -s tcp://192.168.58.193:11001 -b 100
$ python train_mnist_pp_81_dpppmp_3.py -r worker -w 8 -k 7 -g 1 -t "dp+pp+mp" -dpsz 2 -ppsz 2 -mpsz 2 -u tcp://192.168.58.194:12001 -s tcp://192.168.58.193:11001 -b 100

```


## TODO

- [x] DP (Data Parallelism)
- [x] MP (Model Parallelism)
- [x] PP (Pipeline Parallelism)
- [x] ZeRO ([ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf))

