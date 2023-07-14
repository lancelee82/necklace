### necklace examples for PP (Pipeline Model Parallel) training

* pytorch/main.py

MNIST example from pytorch examples [here](https://github.com/pytorch/examples/tree/master/mnist).


#### PP (MP-v)

MP with model vertical split so that different layers is on different nodes.

* pytorch/train_mnist_mp_1_ori.py

MNIST single-machine MP example from pytorch tutorial [here](https://github.com/pytorch/tutorials/blob/master/intermediate_source/model_parallel_tutorial.py).

* pytorch/train_mnist_mp_2_splt.py

MNIST multiple-devices MP example by splitting model to sub-modules and transmitting outputs and gradients between devices.

* pytorch/train_mnist_mp_5_nklc.py

MNIST multiple-devices MP example by necklace.

* pytorch/train_mnist_mp_5_nklc_w3.py

MNIST 3-devices MP example by necklace.


#### DP+PP

