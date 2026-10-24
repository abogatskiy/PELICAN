"""Command-line entry point: pelican {train,eval,info} [options]. See `pelican <command> --help`."""
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from . import __version__
from .config import parse_args, save_config
from .data import build_dataloaders, DATASET_PRESETS
from .models import PelicanModel
from .train import distributed as D
from .train.tasks import make_task
from .train.trainer import Trainer
from .train.ui import setup_logging, UI

logger = logging.getLogger('pelican')


def build_model(cfg):
    return PelicanModel(task=cfg.task, num_classes=cfg.num_classes, num_targets=cfg.num_targets, stabilizer=cfg.stabilizer,
                        method=cfg.method, irc_safe=cfg.irc_safe, num_scalars=DATASET_PRESETS[cfg.dataset][1], scale=cfg.scale,
                        num_channels_scalar=cfg.num_channels_scalar, num_channels_m=cfg.num_channels_m,
                        num_channels_eq=cfg.num_channels_eq, num_channels_m_out=cfg.num_channels_m_out,
                        num_channels_out=cfg.num_channels_out, agg=cfg.agg, agg_out=cfg.agg_out, activation=cfg.activation,
                        activate_agg=cfg.activate_agg, activate_lin=cfg.activate_lin, activate_agg_out=cfg.activate_agg_out,
                        activate_lin_out=cfg.activate_lin_out, factorize=cfg.factorize, batchnorm=cfg.batchnorm,
                        masked=cfg.masked, dropout=cfg.drop_rate, dropout_out=cfg.drop_rate_out, nobj_avg=cfg.nobj_avg,
                        mlp_out=cfg.mlp_out, mlp_out_activate_last=cfg.mlp_out_activate_last)


def select_device(cfg):
    name = cfg.device
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name in ('cuda', 'gpu'):
        if not torch.cuda.is_available():
            raise SystemExit('CUDA requested but not available (use --cpu or --mps)')
        device = torch.device('cuda', max(D.local_rank(), 0))
        if cfg.min_gpu_mem:
            mem = torch.cuda.get_device_properties(device).total_memory // 2 ** 20
            if mem < cfg.min_gpu_mem:
                raise SystemExit(f'GPU has {mem} MB, less than --min-gpu-mem {cfg.min_gpu_mem}')
        return device
    if name == 'mps':
        if cfg.double:
            raise SystemExit('MPS does not support float64; drop --double or use --cpu')
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(cfg):
    if cfg.seed < 0:
        cfg.seed = D.broadcast_object(int(datetime.now().timestamp() * 1e5))
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed % 2 ** 32)
    random.seed(cfg.seed)
    if cfg.reproducible:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def setup(cfg, splits):
    """Everything shared by train/eval: run folder, logging, seed, device, data, model, trainer."""
    rank, world = D.init_distributed()
    main = rank == 0
    if main:
        os.makedirs(cfg.run_dir, exist_ok=True)
    D.barrier()
    console = setup_logging(os.path.join(cfg.run_dir, 'train.log'), main=main)
    logger.info(f'pelican {__version__}: {" ".join(sys.argv)}')
    set_seed(cfg)
    if main and cfg.command != 'eval':
        save_config(cfg, os.path.join(cfg.run_dir, 'config.yaml'))
    device = select_device(cfg)
    dtype = torch.float64 if cfg.double else torch.float32

    loaders = build_dataloaders(cfg, splits, distributed=world > 1)
    model = build_model(cfg).to(device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    ui = UI(console, main=main, log_every=cfg.log_every)
    ui.header(cfg, model, n_params, {s: len(l.dataset) for s, l in loaders.items()}, device)
    return Trainer(cfg, model, make_task(cfg), loaders, device, ui)


def cmd_train(cfg):
    if os.path.exists(os.path.join(cfg.run_dir, 'checkpoint.pt')) and not cfg.load:
        raise SystemExit(f'{cfg.run_dir} already has a checkpoint: pass --load to resume, or choose another --prefix')
    trainer = setup(cfg, ('train', 'valid', 'test'))
    if cfg.self_test and trainer.main:
        from .selftest import run_selftest
        from .train.trainer import _to
        run_selftest(trainer.model, _to(next(iter(trainer.loaders['train'])), trainer.device, trainer.net.dtype), cfg.task)
    best_epoch, best_metrics = trainer.fit()
    results = trainer.evaluate(('test',))
    D.cleanup()
    return best_epoch, best_metrics, results


def cmd_eval(cfg):
    trainer = setup(cfg, ('test',))
    results = trainer.evaluate(('test',))
    D.cleanup()
    return results


def cmd_info(cfg):
    setup_logging(None)
    model = build_model(cfg)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    for name, child in model.named_children():
        print(f'{name:>10}: {sum(p.numel() for p in child.parameters()):,}')
    print(f'{"total":>10}: {total:,} parameters')


def main(argv=None):
    cfg = parse_args(argv)
    try:
        if cfg.command == 'train':
            cmd_train(cfg)
        elif cfg.command == 'eval':
            cmd_eval(cfg)
        elif cfg.command == 'info':
            cmd_info(cfg)
    except ValueError as e:      # configuration errors raised while building the model
        logger.error(str(e))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
