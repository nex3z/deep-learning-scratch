from collections import OrderedDict

import numpy as np

from common.opt import numerical_gradient


class Network(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.layers_cnt = {}
        self.last_layer = None
        self.kernel_regularizer = None

    def add(self, layer, name=None):
        if name is None:
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
            if layer.params is None:
                continue
            for param_name, param in layer.params.items():
                params['{}_{}'.format(layer_name, param_name)] = param
        return params

    @property
    def weights(self):
        params = {}
        for layer_name, layer in self.layers.items():
            if layer.params is None:
                continue
            for param_name, param in layer.params.items():
                if param_name == 'W':
                    params['{}_{}'.format(layer_name, param_name)] = param
        return params

    @property
    def dparams(self):
        dparams = {}
        for layer_name, layer in self.layers.items():
            if layer.dparams is None:
                continue
            for dparam_name, dparam in layer.dparams.items():
                dparams['{}_{}'.format(layer_name, dparam_name)] = dparam
        return dparams

    def build(self, last_layer, kernel_regularizer=None):
        self.last_layer = last_layer
        self.kernel_regularizer = kernel_regularizer

    def predict(self, x, training=False):
        for layer in self.layers.values():
            x = layer.forward(x, training=training)
        return x

    def loss(self, x, y, y_hat=None, training=False):
        if y_hat is None:
            y_hat = self.predict(x, training=training)

        weight_decay = 0
        if self.kernel_regularizer is not None:
            for weight_name, weight in self.weights.items():
                weight_decay += self.kernel_regularizer.loss(weight)

        return self.last_layer.forward(y_hat, y=y) + weight_decay

    def accuracy(self, x, y, y_hat=None):
        if y_hat is None:
            y_hat = self.predict(x, training=False)
        y_hat = np.argmax(y_hat, axis=1)

        if y.ndim != 1:
            y = np.argmax(y, axis=1)

        accuracy = np.sum(y_hat == y) / float(x.shape[0])
        return accuracy

    def evaluate(self, x, y):
        y_hat = self.predict(x)
        loss = self.loss(x, y, y_hat)
        accuracy = self.accuracy(x, y, y_hat)
        return loss, accuracy

    def numerical_gradient(self, x, y):
        def loss(w):
            return self.loss(x, y, training=True)

        grads = {}
        for param_name, param in self.params.items():
            grads[param_name] = numerical_gradient(loss, param)

        return grads

    def gradient(self, x, y):
        self.loss(x, y, training=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        dparams = self.dparams
        if self.kernel_regularizer is not None:
            for weight_name, weight in self.weights.items():
                dparams[weight_name] += self.kernel_regularizer.dloss(weight)

        return dparams
