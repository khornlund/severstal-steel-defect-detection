from torch.optim import lr_scheduler
from scipy.stats import logistic
import numpy as np


class CustomScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer):
        self.n_params = len(optimizer.param_groups)
        super().__init__(optimizer)

    @property
    def lrs(self):
        return self._lrs  # implement me!

    def get_lr(self):
        for i in range(self.n_params):
            yield self._lrs[self.last_epoch - 1]


class WarmupRolloffScheduler(CustomScheduler):

    def __init__(self, optimizer, start_lr, peak_lr, peak_epoch, final_lr, final_epoch):
        self._lrs = self.get_lrs(start_lr, peak_lr, peak_epoch, final_lr, final_epoch)
        super().__init__(optimizer)

    def get_lrs(self, start_lr, peak_lr, peak_epoch, final_lr, final_epoch):
        # warmup from start to peak
        lrs = np.zeros((final_epoch,))
        lrs[0:peak_epoch] = np.linspace(start_lr, peak_lr, peak_epoch)

        # setup rolloff params
        length = final_epoch - peak_epoch
        magnitude = peak_lr - final_lr

        # rolloff to final
        rolloff_lrs = rolloff(length, magnitude=magnitude, offset=final_lr)
        lrs[peak_epoch:] = rolloff_lrs
        return lrs


class CyclicalDecayScheduler(CustomScheduler):

    def __init__(self, optimizer, offset, amplitude, n_periods, n_epochs, gamma):
        self._lrs = self.get_lrs(offset, amplitude, n_periods, n_epochs, gamma)
        super().__init__(optimizer)

    def get_lrs(self, offset, amplitude, n_periods, n_epochs, gamma):
        return sin_decay(offset, amplitude, n_periods, n_epochs, gamma)


class CosineAnnealingScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, start_anneal, n_epochs):
        self.curve = self.get_curve(1, start_anneal, n_epochs)
        self.initial_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    def get_curve(self, start_lr, start_anneal, n_epochs):
        # constant LR to start
        lrs = np.zeros((n_epochs,))
        lrs[0:start_anneal] = start_lr

        # setup rolloff params
        length = n_epochs - start_anneal

        # rolloff to zero
        rolloff_lrs = rolloff(length, loc_factor=0.5, scale_factor=0.1, magnitude=start_lr)
        lrs[start_anneal:] = rolloff_lrs
        return lrs

    def get_lr(self):
        for i, lr in enumerate(self.initial_lrs):
            yield lr * self.curve[self.last_epoch - 1]


# -- Util functions --

def rolloff(length, loc_factor=0.5, scale_factor=0.1, magnitude=1, offset=0):
    """
    Produces a rolloff function over a given length. Imagine 1 - sigmoid(x).
    """
    loc = length * loc_factor
    scale = length * scale_factor
    rolloff = np.array([logistic.sf(x, loc, scale) for x in range(length)])
    rolloff *= magnitude
    rolloff += offset
    return rolloff


def sin_decay(offset, amplitude, n_periods, n_epochs, gamma):
    """
    Produces a sinusoidal decay function.
    """
    max_x = n_periods * 2 * np.pi
    xs = np.linspace(0, max_x, n_epochs)
    sin = np.sin(xs)
    gammas = np.array([gamma ** x for x in range(n_epochs)])
    sin *= gammas
    sin -= (1 - gammas)
    sin += 1
    sin *= amplitude / 2
    sin += offset
    return sin
