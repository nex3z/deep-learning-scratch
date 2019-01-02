import numpy as np
from common.function import sigmoid, softmax, cross_entropy_error
from common.initializer import normal, zeros
from common.exception import UnimplementedMethodException


class BaseLayer(object):
    def backward(self, dout):
        raise UnimplementedMethodException()

    @property
    def params(self):
        return None

    @property
    def dparams(self):
        return None


class Add(object):
    @staticmethod
    def forward(x, y):
        return x + y

    @staticmethod
    def backward(dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Multiply(object):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class Dense(BaseLayer):
    def __init__(self, input_size, output_size, kernel_initializer=normal, bias_initializer=zeros):
        self.W = kernel_initializer(input_size, output_size)
        self.b = bias_initializer(output_size)
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = dout.dot(self.W.T)
        self.dW = self.x.T.dot(dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

    @property
    def params(self):
        return {
            'W': self.W,
            'b': self.b,
        }

    @property
    def dparams(self):
        return {
            'W': self.dW,
            'b': self.db,
        }


class Relu(BaseLayer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(BaseLayer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SoftmaxWithLoss(BaseLayer):
    def __init__(self):
        self.loss = None
        self.y_hat = None
        self.y = None

    def forward(self, x, y):
        self.y = y
        self.y_hat = softmax(x)
        self.loss = cross_entropy_error(self.y_hat, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) / batch_size
        return dx


class Dropout(BaseLayer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training):
        if training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)

    def backward(self, dout):
        return dout * self.mask
