### necklace examples for MP (Tensor Model Parallel) training

* pytorch/main.py

MNIST example from pytorch examples [here](https://github.com/pytorch/examples/tree/master/mnist).


#### MP (MP-h)

MP with model horizontal split so that different parts of a large layer is on different nodes.

* pytorch/train_mnist_mpnn_21_one_grp.py

MNIST example for MP-h with only one MP group and without DP training. 

* pytorch/train_mnist_mpnn_22_mpdp_grps.py

MNIST example for MP-h with multiple MP groups and with multiple DP groups training. 
