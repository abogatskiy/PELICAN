# Source: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_epochs: target learning rate is reached at warmup_epochs, gradually
        after_scheduler: after warmup_epochs, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_epochs, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.warmup_epochs:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epochs)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class GradualCooldownScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        cooldown_epoch: epoch of Cosine Annealing at which the exponential LR decay kicks in (counted from the end of warmup, not from the beginning!)
        after_scheduler: original scheduler that works until cooldown_epoch (eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, lr_final, cooldown_epoch, cooldown_length, after_scheduler):
        self.lr_final = lr_final
        self.cooldown_epoch = cooldown_epoch
        self.cooldown_length = cooldown_length
        self.after_scheduler = after_scheduler
        self.started = False
        super(GradualCooldownScheduler, self).__init__(optimizer)

    def get_lr(self):
        if (not self.started) and self.after_scheduler.last_epoch < self.cooldown_epoch:
            return self.after_scheduler.get_last_lr()
        return [init_lr * (self.lr_final/init_lr) ** (self.last_epoch/self.cooldown_length) for init_lr in self.init_cooldown_lr]

    def step(self, epoch=None, metrics=None):
        if self.last_epoch == -1:
            return super(GradualCooldownScheduler, self).step(epoch)
        if hasattr(self.after_scheduler, 'after_scheduler'):
            if self.after_scheduler.after_scheduler.last_epoch >= self.cooldown_epoch - 1:
                self.started = True
        else:
            if self.after_scheduler.last_epoch >= self.cooldown_epoch - 1:
                self.started = True

        if (not self.started) and self.after_scheduler:
            self.after_scheduler.step(epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
            self.init_cooldown_lr = self._last_lr
        else:
            return super(GradualCooldownScheduler, self).step(epoch)