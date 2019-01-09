import numpy as np
from common.function import sigmoid, softmax, cross_entropy_error
from common.initializer import normal, zeros
from common.exception import UnimplementedMethodException
from common.util import col2im, im2col


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
        self.loss = cross_entropy_error(self.y, self.y_hat)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        is_one_hot = self.y.size == self.y_hat.size
        if is_one_hot:
            dx = (self.y_hat - self.y) / batch_size
        else:
            dx = self.y_hat.copy()
            dx[np.arange(batch_size), self.y] -= 1
            dx = dx / batch_size
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


class Conv2D(BaseLayer):
    def __init__(self, input_size, filters, kernel_size=(3, 3), stride=1, padding=0, kernel_initializer=normal,
                 bias_initializer=zeros):
        self.W = kernel_initializer(filters, input_size[0], *kernel_size)
        self.b = bias_initializer(filters)
        self.dW = None
        self.db = None

        self.stride = stride
        self.pad = padding

        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x, **kwargs):
        kernel_n, kernel_c, kernel_h, kernel_w = self.W.shape
        x_n, x_c, x_h, x_w = x.shape
        out_h = 1 + int((x_h + 2 * self.pad - kernel_h) / self.stride)
        out_w = 1 + int((x_w + 2 * self.pad - kernel_w) / self.stride)

        col = im2col(x, kernel_h, kernel_w, self.stride, self.pad)
        col_W = self.W.reshape(kernel_n, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(x_n, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        kernel_n, kernel_c, kernel_h, kernel_w = self.W.shape
        dout = dout.transpose((0, 2, 3, 1)).reshape(-1, kernel_n)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose((1, 0)).reshape(kernel_n, kernel_c, kernel_h, kernel_w)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, kernel_h, kernel_w, self.stride, self.pad)

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


class MaxPool2D(BaseLayer):
    def __init__(self, pool_size, stride=1, padding=0):
        self.pool_h, self.pool_w = pool_size
        self.stride = stride
        self.pad = padding

        self.x = None
        self.arg_max = None

    def forward(self, x, **kwargs):
        n, c, h, w = x.shape
        out_h = int(1 + (h - self.pool_h) / self.stride)
        out_w = int(1 + (w - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(n, out_h, out_w, c).transpose((0, 3, 1, 2))

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose((0, 2, 3, 1))

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
