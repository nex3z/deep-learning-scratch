from collections import OrderedDict

import numpy as np

from basic.initializer import normal, xavier, he, zeros
from basic.layer import Dense, Relu, Sigmoid, SoftmaxWithLoss
from basic.opt import numerical_gradient


class MultiLayerNet(object):
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std=0.01,
                 weight_decay_lambda=0):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.num_hidden_layer = len(hidden_size_list)
        self.output_size = output_size
        self.activation = activation
        self.weight_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.__init_weight()

        self.layers = OrderedDict()
        activation_layer = Relu if self.activation == 'relu' else Sigmoid
        for idx in range(1, self.num_hidden_layer + 1):
            self.layers['Dense{}'.format(idx)] = Dense(self.params['W{}'.format(idx)], self.params['b{}'.format(idx)])
            self.layers['ActivateFunction{}'.format(idx)] = activation_layer()

        idx = self.num_hidden_layer + 1
        self.layers['Dense{}'.format(idx)] = Dense(self.params['W{}'.format(idx)], self.params['b{}'.format(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if str(self.activation).lower() == 'sigmoid':
                weight = xavier(all_size_list[idx - 1], all_size_list[idx])
            elif str(self.activation).lower() == 'relu':
                weight = he(all_size_list[idx - 1], all_size_list[idx])
            else:
                weight = normal(all_size_list[idx - 1], all_size_list[idx], std=self.weight_init_std)
            self.params['W{}'.format(idx)] = weight
            self.params['b{}'.format(idx)] = zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.num_hidden_layer + 2):
            W = self.params['W{}'.format(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.num_hidden_layer + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.num_hidden_layer + 2):
            grads['W' + str(idx)] = self.layers['Dense' + str(idx)].dW + self.weight_decay_lambda * self.layers['Dense' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Dense' + str(idx)].db

        return grads
