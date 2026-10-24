"""Optimizers and the learning-rate schedule."""
import math
import torch


def make_optimizer(cfg, model):
    params = [{'params': model.parameters(), 'lr': cfg.lr_init, 'weight_decay': cfg.weight_decay}]
    name = cfg.optim.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params)
    if name == 'adam':
        return torch.optim.Adam(params)
    if name == 'amsgrad':
        return torch.optim.Adam(params, amsgrad=True)
    if name == 'radam':
        return torch.optim.RAdam(params)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params)
    if name == 'sgd':
        return torch.optim.SGD(params, nesterov=True, momentum=0.2)
    raise ValueError(f'Unknown optimizer {cfg.optim}')


def lr_factor(step, steps_per_epoch, cfg):
    """
    Multiplier of lr_init at a given optimizer step: linear warmup over `warmup` epochs, then the main schedule,
    then an exponential cooldown (halving every epoch, floored at lr_final) over the last `cooldown` epochs.
    Main schedules: 'cos' anneals to lr_final; 'warm' is cosine with warm restarts after 4, 8, 16, ... epochs;
    'flat' is constant; 'exp' decays exponentially to lr_final.
    """
    spe = steps_per_epoch
    if cfg.lr_per_epoch:
        step = (step // spe) * spe
    warm, cool = cfg.warmup * spe, cfg.cooldown * spe
    main = max(cfg.num_epoch * spe - warm - cool, 1)
    r = cfg.lr_final / cfg.lr_init

    def main_factor(s):
        s = min(s, main)
        if cfg.lr_schedule == 'flat':
            return 1.0
        if cfg.lr_schedule == 'exp':
            return r ** (s / main)
        if cfg.lr_schedule == 'cos':
            return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * s / main))
        if cfg.lr_schedule == 'warm':
            period = 4 * spe
            while s >= period:
                s -= period
                period *= 2
            return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * s / period))
        raise ValueError(f'Unknown lr schedule {cfg.lr_schedule}')

    if step < warm:
        return (step + 1) / warm
    if step < warm + main:
        return main_factor(step - warm)
    start = main_factor(main)
    return max(r, start * 0.5 ** ((step - warm - main) / spe))


def make_scheduler(cfg, optimizer, steps_per_epoch):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, steps_per_epoch, cfg))
