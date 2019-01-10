import matplotlib.pyplot as plt

from data.BatchGenerator import BatchGenerator
from net.Network import Network
from optimizer.GradientDescent import GradientDescent
from common.ProgressBar import ProgressBar


class Model(object):
    def __init__(self):
        self.network = Network()
        self.optimizer = None

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def add(self, layer, name=None):
        self.network.add(layer, name)

    def build(self, last_layer, optimizer=GradientDescent(), kernel_regularizer=None):
        self.network.build(last_layer, kernel_regularizer)
        self.optimizer = optimizer

    def fit(self, x, y, batch_size, epochs, validation_data=None, eval_sample_num=None, verbose=0):
        batch_generator = BatchGenerator(x, y, batch_size)
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            pb = ProgressBar(max_val=batch_generator.batch_per_epoch) if verbose >= 1 else None
            for batch in range(batch_generator.batch_per_epoch):
                if pb:
                    pb.update(batch + 1)
                x_batch, y_batch = batch_generator.next_batch()
                grads = self.network.gradient(x_batch, y_batch)
                self.optimizer.update(self.network.params, grads)
            if pb:
                pb.finish()

            train_loss, train_acc = self.network.evaluate(x[:eval_sample_num], y[:eval_sample_num])
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if validation_data:
                (val_x, val_y) = validation_data
                val_loss, val_acc = self.network.evaluate(val_x[:eval_sample_num], val_y[:eval_sample_num])
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                print("train_loss = {:.4f}, train_acc = {:.4f}, val_loss = {:.4f}, val_acc = {:.4f}"
                      .format(train_loss, train_acc, val_loss, val_acc))
            else:
                print("train_loss = {:.4f}, train_acc = {:.4f}".format(train_loss, train_acc))

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
