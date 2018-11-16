from collections import OrderedDict

import numpy as np

from functions.diff import numerical_gradient
from layers.Dense import Dense
from layers.Relu import Relu
from layers.SoftmaxWithLoss import SoftmaxWithLoss


class TwoLayerNet2(object):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

        self.layers = OrderedDict()
        self.layers['dense_1'] = Dense(self.params['W1'], self.params['b1'])
        self.layers['relu_1'] = Relu()
        self.layers['dense_2'] = Dense(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = 1.0 * np.sum(y == t) / x.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2'])
        }
        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {'W1': self.layers['dense_1'].dW,
                 'b1': self.layers['dense_1'].db,
                 'W2': self.layers['dense_2'].dW,
                 'b2': self.layers['dense_2'].db
                 }

        return grads

