import numpy as np
from common.function import sigmoid, softmax, cross_entropy_error
from common.initializer import normal, zeros
from common.exception import UnimplementedMethodException


class BaseLayer(object):
    def forward(self, x, **kwargs):
        raise UnimplementedMethodException()

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

    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
        self.y = kwargs['y']
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

    def forward(self, x, **kwargs):
        if kwargs['training']:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)

    def backward(self, dout):
        return dout * self.mask


class BatchNorm(BaseLayer):
    def __init__(self, input_size, momentum=0.9):
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.dgamma = None
        self.dbeta = None

        self.momentum = momentum
        self.running_mean = None
        self.running_var = None

        self.input_shape = None
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None

    def forward(self, x, **kwargs):
        self.input_shape = x.shape
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        out = self.__forward(x, kwargs['training'])
        return out.reshape(*self.input_shape)

    def __forward(self, x, training):
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])

        if training:
            xc, xn, mean, var, std = self.__normalize(x)
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.std = std
        else:
            xc, xn, _, _, _ = self.__normalize(x, self.running_mean, self.running_var)

        out = self.gamma * xn + self.beta
        return out

    @staticmethod
    def __normalize(x, mean=None, var=None):
        mean = np.mean(x, axis=0) if mean is None else mean
        xc = x - mean
        var = np.mean(xc ** 2, axis=0) if var is None else var
        std = np.sqrt(var + 10e-7)
        xn = xc / std
        return xc, xn, mean, var, std

    def backward(self, dout):
        if dout.ndim != 2:
            dout = dout.reshape(dout.shape[0], -1)
        dx = self.__backward(dout)
        return dx.reshape(*self.input_shape)

    def __backward(self, dout):
        self.dbeta = dout.sum(axis=0)
        self.dgamma = np.sum(self.xn * dout, axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        return dx

    @property
    def params(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta,
        }

    @property
    def dparams(self):
        return {
            'gamma': self.dgamma,
            'beta': self.dbeta,
        }
