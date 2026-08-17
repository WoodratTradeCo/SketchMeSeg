import os
import time


class Writer:
    """Prints progress to the console and optionally appends it to a log file."""

    def __init__(self, opt):
        self.expr_dir = os.path.join('logs', opt.dataset, opt.class_name,
                                     time.strftime('%b%d_%H_%M_%S'))
        self.log = opt.log
        self.log_name = os.path.join(self.expr_dir, 'log.txt')
        if self.log:
            os.makedirs(self.expr_dir, exist_ok=True)

    def _write(self, message):
        print(message)
        if self.log:
            with open(self.log_name, 'a') as f:
                f.write(message + '\n')

    def print_train_loss(self, epoch, i, loss):
        self._write('(time: %s, epoch: %d, iters: %d) loss: %.3f' %
                    (time.strftime('%X %x'), epoch, i, loss.item()))

    def print_epoch_train_loss(self, loss):
        self._write('epoch_loss: %.3f' % loss)

    def print_test_result(self, epoch, ave_loss, p, c, best_P):
        self._write('epoch: %d, TEST: Ave_Loss %.5f P_metric %.3f%% '
                    'C_metric %.3f%%, BEST_P %.3f%%' %
                    (epoch, ave_loss, p * 100, c * 100, best_P * 100))
        self._write('====================================================')