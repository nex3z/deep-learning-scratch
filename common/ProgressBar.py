import math
import sys


class ProgressBar(object):
    def __init__(self, max_val=100, bar_size=30, update_step=1):
        self.max_val = max_val
        self.bar_size = bar_size
        self.update_step = update_step
        self.progress = 0
        self.pattern = '{{:{:d}d}}/{:d} [{{:{:d}s}}] {{:.1f}}% {{}}'.format(len(str(max_val)), max_val, self.bar_size)

    def update(self, progress, prefix='', postfix=''):
        if progress > self.max_val:
            return
        elif progress != self.max_val and abs(progress - self.progress) < self.update_step:
            return

        num_bar = math.floor(progress / self.max_val * self.bar_size)

        msg = self.pattern.format(progress, '=' * num_bar, progress / self.max_val * 100, postfix)
        if prefix:
            msg = prefix + ' ' + msg

        sys.stdout.write('\r')
        sys.stdout.write(msg)
        sys.stdout.flush()

        self.progress = progress

    @staticmethod
    def finish():
        sys.stdout.write('\n')
        sys.stdout.flush()
