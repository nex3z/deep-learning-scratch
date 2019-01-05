import math

import matplotlib.pyplot as plt
import numpy as np

from net.Network import Network
from optimizer.GradientDescent import GradientDescent


class Model(object):
    def __init__(self):
        self.network = Network()
        self.optimizer = None
        self.iter_cnt = 0

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def add(self, layer, name=None):
        self.network.add(layer, name)

    def build(self, last_layer, optimizer=GradientDescent(), weight_decay_lambda=0):
        self.network.build(last_layer, weight_decay_lambda)
        self.optimizer = optimizer

    def fit(self, x, y, batch_size, epochs, validation_data=None):
        train_size = y.shape[0]
        batch_per_epoch = math.ceil(train_size / batch_size)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs + 1))
            for batch in range(batch_per_epoch):
                self.iter_cnt += 1
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x[batch_mask]
                y_batch = y[batch_mask]

                grads = self.network.gradient(x_batch, y_batch)
                self.optimizer.update(self.network.params, grads)

            train_loss, train_acc = self.network.evaluate(x, y)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if validation_data:
                (val_x, val_y) = validation_data
                val_loss, val_acc = self.network.evaluate(val_x, val_y)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                print("Iter: {}, train_loss = {:.4f}, train_acc = {:.4f}, val_loss = {:.4f}, val_acc = {:.4f}"
                      .format(self.iter_cnt, train_loss, train_acc, val_loss, val_acc))
            else:
                print("Iter: {}, train_loss = {:.4f}, train_acc = {:.4f}".format(self.iter_cnt, train_loss, train_acc))

    def plot_history(self, metric='loss'):
        if metric == 'loss':
            train_history = self.train_loss_history
            val_history = self.val_loss_history
        elif metric == 'accuracy':
            train_history = self.train_acc_history
            val_history = self.val_acc_history
        else:
            return
        plt.figure()
        epochs = range(1, len(train_history) + 1)
        plt.plot(epochs, train_history, marker='o', label='train')
        plt.plot(epochs, val_history, marker='o', label='validation')
        plt.legend()
        plt.show()
