#!/bin/bash

n=3
b=0
a=8

if [ $# -gt 0 ] ; then
    n=$1
fi

if [ $# -gt 1 ] ; then
    b=$2
fi

if [ $# -gt 2 ] ; then
    a=$3
fi

for ((i=0; i<$n; i++))
{
    # scheduler:: python train_ngn_mnist_1.py -r scheduler -w 2 -k 0 -e 3 -u tcp://10.60.242.136:9399

    echo $i

    #python train_ngn_mnist_1.py -r worker -w $a -k $i -u tcp://10.60.242.136:930$i -s tcp://10.60.242.136:9399 --gpus $i &

    # for multi nodes
    kk=`expr $b + $i`
    python train_ngn_mnist_1.py -r worker -w $a -k ${kk} -u tcp://10.60.242.136:930$i -s tcp://10.60.242.136:9399 --gpus $i --lr 0.05 &

    gg=`expr 1 + $i`
    #python train_ngn_mnist_1.py -r worker -w $a -k ${kk} -u tcp://10.60.242.136:930$i -s tcp://10.60.242.136:9399 --gpus ${gg} &
}
