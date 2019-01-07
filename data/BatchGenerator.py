import math

import numpy as np


class BatchGenerator(object):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.data_size = y.shape[0]
        self.batch_per_epoch = math.floor(self.data_size / batch_size)
        self.perm = []
        self.perm_index = -1

        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.data_size)
        np.random.shuffle(perm)
        self.perm = perm
        self.perm_index = 0

    def next_batch(self):
        if self.perm_index + self.batch_size > self.data_size:
            self.shuffle()

        start = self.perm_index
        end = start + self.batch_size
        self.perm_index += self.batch_size

        indexes = self.perm[start:end]
        return self.x[indexes], self.y[indexes]
