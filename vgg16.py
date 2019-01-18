#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cupy as xp
import chainer
from chainer import cuda, Function, \
    report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import datasets
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L


class my_VGG16(Chain):
    def __init__(self, num_class):
        '''
        num_class : 分類するクラスの数
        '''
        super(my_VGG16, self).__init__(
            base=L.VGG16Layers(pretrained_model='auto'),
            fc8=L.Linear(None, num_class)
        )

    def __call__(self, x):
        '''
        学習
        '''
        x = self.base(x, layers=['fc7'])
        x = self.fc8(x['fc7'])
        return x


def train_for_vgg16(train):
    train_dataset = []
    for x_train, y_train in train:
        x_train = chainer.links.model.vision.vgg.prepare(x_train)
        y_train = xp.int32(y_train)
        train_dataset.append((x_train, y_train))
    return train_dataset


def model_train(model, train, optimizer, epoch=10, batch_size=50):
    gpu = 1 
    cuda.get_device(gpu).use()
    split_at = int(len(train) * 0.9)
    train, val = chainer.datasets.split_dataset(train, split_at)
    train_iterator = iterators.SerialIterator(train, batch_size)
    val_iterator = iterators.SerialIterator(val, batch_size, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iterator, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(val_iterator, model, device=gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.snapshot())
    trainer.run()


def main():
    # データの読み込み
    train, test = datasets.get_mnist(ndim=3, dtype='float32')

    train_dataset = train_for_vgg16(train)

    # モデルの読み込み
    model = L.Classifier(my_VGG16(num_class=10))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # pre_trainしたモデルの重みの更新を行わないようにする
    model.predictor.base.disable_update()

    model_train(model, train_dataset, optimizer, epoch=10, batch_size=50)

    acc = 0
    for x, y in test[:20]:
        # img = Image.open(x)
        x = chainer.links.model.vision.vgg.prepare(x)
        x = x[xp.newaxis]
        out = model.predictor(x)
        print(out.data)
        print(out.data.max())
        print(np.argmax(out.data))
        if np.argmax(out.data) == y:
            print('ok', y)
            acc += 1

    print(acc/len(test[:20])*100)

if __name__ == '__main__':
    main()
