from collections import OrderedDict

import numpy as np

from common.opt import numerical_gradient
from optimizer.GradientDescent import GradientDescent


class Network(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.layers_cnt = {}
        self.last_layer = None
        self.optimizer = None

    def add(self, layer, name=None):
        if not name:
            name = self.__get_name_for_layer(layer)
        self.layers[name] = layer

    def __get_name_for_layer(self, layer):
        layer_name = type(layer).__name__
        layer_number = 1 if layer_name not in self.layers_cnt else self.layers_cnt[layer_name] + 1
        self.layers_cnt[layer_name] = layer_number
        name = '{}_{}'.format(layer_name, layer_number)
        return name

    @property
    def params(self):
        params = {}
        for layer_name, layer in self.layers.items():
            if not layer.params:
                continue
            for param_name, param in layer.params.items():
                params['{}_{}'.format(layer_name, param_name)] = param
        return params

    def build(self, last_layer, optimizer=GradientDescent):
        self.last_layer = last_layer
        self.optimizer = optimizer

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        y_hat = self.predict(x)
        return self.last_layer.forward(y_hat, y)

    def accuracy(self, x, y):
        y_hat = self.predict(x)
        y_hat = np.argmax(y_hat, axis=1)
        if y.ndim != 1:
            y = np.argmax(y, axis=1)

        accuracy = np.sum(y_hat == y) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, y):
        def loss(w):
            return self.loss(x, y)

        grads = {}
        for layer_name, layer in self.layers.items():
            if not layer.params:
                continue
            for param_name, param in layer.params.items():
                grads['{}_{}'.format(layer_name, param_name)] = numerical_gradient(loss, param)

        return grads

    def gradient(self, x, y):
        self.loss(x, y)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for layer_name, layer in self.layers.items():
            if not layer.dparams:
                continue
            for param_name, param in layer.dparams.items():
                grads['{}_{}'.format(layer_name, param_name)] = param

        return grads
