#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cupy as xp
import os
import chainer
from chainer import cuda, Function, \
    report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import datasets
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
from vgg16 import my_VGG16


def main():
    # データの読み込み
    _, test = datasets.get_mnist(ndim=3, dtype='float32')

    # モデルの読み込み
    model = L.Classifier(my_VGG16(num_class=10))

    serializers.load_npz('result/snapshot_iter_10800', model, strict=False)

    acc = 0
    for x, y in test:
        # img = Image.open(x)
        x = chainer.links.model.vision.vgg.prepare(x)
        x = x[xp.newaxis]
        out = model.predictor(x)
        if np.argmax(out.data) == y:
            acc += 1

    print('test :', len(test))
    print(acc/len(test)*100)

if __name__ == '__main__':
    main()
