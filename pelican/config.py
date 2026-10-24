"""
Run configuration: command-line arguments, YAML layering and validation.

Precedence (lowest to highest): built-in defaults < YAML files given with --config (in order) < command-line flags.
For `pelican eval`, the run's saved config.yaml is applied before any --config files.
YAML keys are the argument names with underscores (e.g. num_channels_m: [[60], [60]]).
"""
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

from .symmetry import STABILIZERS
from .nn.equivariant import AGGREGATIONS
from .data.collate import DATASET_PRESETS

logger = logging.getLogger(__name__)

# Keys of the pre-package scripts (train_pelican_*.py) that were renamed, and ones that no longer exist.
RENAMED = {'config': 'agg', 'config_out': 'agg_out', 'num_channels_2to2': 'num_channels_eq', 'lr_decay_type': 'lr_schedule',
           'RAMdataset': 'ram_dataset', 'yaml': 'config', 'test': 'self_test', 'predict': 'save_predictions'}
REMOVED = {'rank1_width_multiplier', 'dropout', 'quiet', 'textlog', 'verbose', 'alpha', 'lr_decay', 'sgd_restart', 'read_pid',
           'force_download', 'weight_init', 'summarize_csv', 'workdir', 'logdir', 'modeldir', 'predictdir', 'loadfile',
           'checkfile', 'bestfile', 'logfile', 'predictfile', 'lr_minibatch', 'log_level', 'save', 'dtype', 'add_beams',
           'sig', 'softmasked', 'ir_safe', 'c_safe', 'host', 'port', 'password', 'storage', 'sampler', 'pruner', 'study_name',
           'optuna_test', 'n_trials', 'metric'}
# Internal keys that are never read from YAML.
_INTERNAL = {'command', 'config', 'run'}


def _add_arguments(p):
    BOOL = argparse.BooleanOptionalAction
    p.add_argument('--config', '--yaml', action='append', default=None, metavar='YAML',
                   help='YAML file(s) with argument values; later files override earlier ones')

    g = p.add_argument_group('run')
    g.add_argument('--prefix', default=None, help='run name; outputs go to <runs-dir>/<prefix>/ (default: pelican-<timestamp>)')
    g.add_argument('--runs-dir', default='runs', help='parent directory of run folders (default: runs)')
    g.add_argument('--load', action=BOOL, default=False, help='resume training from the run folder\'s checkpoint')
    g.add_argument('--seed', type=int, default=-1, help='random seed (-1: from the clock)')
    g.add_argument('--reproducible', action=BOOL, default=False, help='deterministic algorithms (slower on GPU)')
    g.add_argument('--device', default='auto', help='auto | cuda | mps | cpu')
    g.add_argument('--cpu', dest='device', action='store_const', const='cpu', help='shortcut for --device cpu')
    g.add_argument('--cuda', '--gpu', dest='device', action='store_const', const='cuda', help='shortcut for --device cuda')
    g.add_argument('--mps', dest='device', action='store_const', const='mps', help='shortcut for --device mps')
    g.add_argument('--double', action=BOOL, default=False, help='float64 model and inputs')
    g.add_argument('--num-workers', type=int, default=0, help='dataloader worker processes')
    g.add_argument('--distribute-eval', action=BOOL, default=False, help='split test-set evaluation across GPUs (order not preserved)')
    g.add_argument('--min-gpu-mem', type=int, default=0, help='exit if the GPU has less memory than this (MB)')
    g.add_argument('--log-every', type=int, default=0, help='also log every N minibatches (0: per epoch only)')
    g.add_argument('--save-every', type=int, default=0, help='also checkpoint every N minibatches')
    g.add_argument('--save-predictions', action=BOOL, default=True, help='save test-set predictions and ROC curves')
    g.add_argument('--summarize', action=BOOL, default=False, help='write TensorBoard summaries')
    g.add_argument('--self-test', action=BOOL, default=False, help='check symmetries on one batch before training')

    g = p.add_argument_group('data')
    g.add_argument('--datadir', default='data/', help='directory with *train*.h5, *valid*.h5, *test*.h5')
    g.add_argument('--testfile', default='', help='evaluate on this file instead of the test files in datadir')
    g.add_argument('--dataset', default='', choices=list(DATASET_PRESETS), help="per-particle feature preset ('jc' = JetClass)")
    g.add_argument('--target', default='is_signal', help='HDF5 key of the training target')
    g.add_argument('--num-classes', type=int, default=2, help='number of classes (classification)')
    g.add_argument('--num-targets', type=int, default=1, help='number of target 4-vectors (regression)')
    g.add_argument('--nobj', type=int, default=None, help='keep at most this many constituents per event')
    g.add_argument('--nobj-avg', type=int, default=49, help='typical multiplicity; normalizes sums and the nobj^alpha scalings')
    g.add_argument('--num-train', type=int, default=-1, help='training events (-1: all)')
    g.add_argument('--num-valid', type=int, default=-1, help='validation events (-1: all)')
    g.add_argument('--num-test', type=int, default=-1, help='test events (-1: all)')
    g.add_argument('--ram-dataset', action=BOOL, default=True, help='load datasets into RAM')
    g.add_argument('--shuffle', action=BOOL, default=True, help='shuffle training batches')
    g.add_argument('--balance', action=BOOL, default=True, help='interleave classes (binary classification)')
    g.add_argument('--fix-data', action=BOOL, default=False, help='fixed seed for choosing the event subsets')
    g.add_argument('--scale', type=float, default=1.0, help='multiply input 4-momenta by this factor')

    g = p.add_argument_group('model')
    g.add_argument('--task', default='classification', choices=['classification', 'regression'])
    g.add_argument('--stabilizer', default='so2', choices=list(STABILIZERS), help='residual symmetry group')
    g.add_argument('--method', default='spurions', choices=['spurions', 'input'],
                   help='symmetry breaking: spurion particles, or stabilizer-invariant components in the pairwise inputs')
    g.add_argument('--irc-safe', action=BOOL, default=False, help='IRC-safe model (energy-weighted aggregations, angle inputs)')
    g.add_argument('--num-channels-scalar', type=int, default=10, help='pair channels produced from the particle scalars (spurion labels, --dataset features)')
    g.add_argument('--num-channels-m', nargs='*', type=int, default=[[60]] * 5,
                   help='hidden widths of each block\'s MessageNet; in YAML a list of lists, one per block')
    g.add_argument('--num-channels-eq', nargs='*', type=int, default=[35] * 5, help='input width of each equivariant block')
    g.add_argument('--num-channels-m-out', nargs='*', type=int, default=[60, 35], help='final message layer widths')
    g.add_argument('--num-channels-out', nargs='*', type=int, default=[60], help='output MLP widths')
    g.add_argument('--agg', default='M', help=f'aggregation in the equivariant blocks, one of {AGGREGATIONS}')
    g.add_argument('--agg-out', default='M', help='aggregation in the pooling layer (and in lifting the particle scalars)')
    g.add_argument('--activation', default='leakyrelu')
    g.add_argument('--activate-agg', action=BOOL, default=False, help='activation right after equivariant aggregation')
    g.add_argument('--activate-lin', action=BOOL, default=True, help='activation after the equivariant linear map')
    g.add_argument('--activate-agg-out', action=BOOL, default=True)
    g.add_argument('--activate-lin-out', action=BOOL, default=False)
    g.add_argument('--factorize', action=BOOL, default=True, help='low-rank equivariant coefficients')
    g.add_argument('--batchnorm', action=BOOL, default=True, help='batch norm after each MessageNet')
    g.add_argument('--masked', action=BOOL, default=True, help='batch norm ignores padding')
    g.add_argument('--drop-rate', type=float, default=0.025, help='dropout inside the network (0 disables)')
    g.add_argument('--drop-rate-out', type=float, default=0.025, help='dropout before the output MLP')
    g.add_argument('--mlp-out', action=BOOL, default=True, help='output MLP after pooling (needed for regression)')
    g.add_argument('--mlp-out-activate-last', action=BOOL, default=True,
                   help='activation after the last layer of the output MLP too, i.e. on the outputs, as in models trained before this option existed')

    g = p.add_argument_group('training')
    g.add_argument('--num-epoch', type=int, default=35)
    g.add_argument('--batch-size', '-bs', type=int, default=128)
    g.add_argument('--optim', default='adamw', choices=['adamw', 'adam', 'amsgrad', 'radam', 'rmsprop', 'sgd'])
    g.add_argument('--lr-init', type=float, default=0.0025)
    g.add_argument('--lr-final', type=float, default=1e-5)
    g.add_argument('--lr-schedule', default='warm', choices=['warm', 'cos', 'flat', 'exp'],
                   help='warm: cosine with warm restarts (4, 8, 16, ... epochs); cos; flat; exp')
    g.add_argument('--lr-per-epoch', action=BOOL, default=False, help='update the LR once per epoch instead of per step')
    g.add_argument('--warmup', type=int, default=4, help='epochs of linear LR warmup')
    g.add_argument('--cooldown', type=int, default=3, help='epochs of exponential LR cooldown at the end')
    g.add_argument('--weight-decay', type=float, default=0.005)
    g.add_argument('--clip-grad', type=float, default=0.0, help='clip the gradient norm to this value (0: off)')
    g.add_argument('--loss', default='m:0.05,3d:0.01', help='regression loss as name:weight pairs')


def build_parser():
    parser = argparse.ArgumentParser(prog='pelican', description='PELICAN: Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network')
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    _add_arguments(common)
    sub.add_parser('train', parents=[common], help='train a model, then evaluate it on the test set')
    ev = sub.add_parser('eval', parents=[common], help='evaluate a trained run on the test set')
    ev.add_argument('run', nargs='?', default=None, help='run folder (default: <runs-dir>/<prefix>)')
    sub.add_parser('info', parents=[common], help='print the model and its parameter count')
    return parser


def read_yaml(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    out = {}
    for key, value in data.items():
        if key == 'task' and value in ('train', 'eval'):     # old meaning; now the `pelican train/eval` command
            logger.warning(f"{path}: ignoring obsolete 'task: {value}' (use the train/eval command)")
            continue
        if key in REMOVED:
            logger.warning(f"{path}: ignoring obsolete key '{key}'")
            continue
        if key in RENAMED:
            logger.warning(f"{path}: key '{key}' is now '{RENAMED[key]}'")
            key = RENAMED[key]
        out[key] = value
    return out


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    sub = parser._subparsers._group_actions[0].choices[args.command]
    valid = {a.dest for a in sub._actions}
    layers = []
    if args.command == 'eval':
        run = args.run or os.path.join(args.runs_dir, args.prefix or '')
        saved = os.path.join(run, 'config.yaml')
        if not os.path.exists(saved):
            raise SystemExit(f'pelican eval: no config.yaml in {run}')
        layers.append(saved)
    layers += args.config or []
    for path in layers:
        values = read_yaml(path)
        unknown = set(values) - valid
        if unknown:
            raise SystemExit(f"{path}: unknown key(s): {', '.join(sorted(unknown))}")
        sub.set_defaults(**{k: v for k, v in values.items() if k not in _INTERNAL})
    args = parser.parse_args(argv)
    if args.command == 'eval' and args.run:
        args.runs_dir, args.prefix = os.path.split(os.path.normpath(args.run))
    return validate(args)


def validate(cfg):
    L = len(cfg.num_channels_eq)
    m = cfg.num_channels_m
    if len(m) == 0:
        cfg.num_channels_m = [[] for _ in range(L)]
    elif all(isinstance(x, int) for x in m):     # flat list on the command line: the same MessageNet in every block
        cfg.num_channels_m = [list(m) for _ in range(L)]
    if len(cfg.num_channels_m) != L:
        raise SystemExit(f'num_channels_m has {len(cfg.num_channels_m)} blocks but num_channels_eq has {L}')
    if isinstance(cfg.batchnorm, str):              # old YAML/checkpoints: 'b' (batch norm) or 'False'/'None'
        cfg.batchnorm = cfg.batchnorm.lower() not in ('false', 'none', '0', '')
    for key in ('agg', 'agg_out'):
        v = getattr(cfg, key)
        if len(v) != 1 or v not in AGGREGATIONS:
            raise SystemExit(f"--{key.replace('_', '-')} must be a single letter from '{AGGREGATIONS}', got '{v}'")
    if cfg.task == 'regression' and not cfg.mlp_out:
        raise SystemExit('regression needs the output MLP (--mlp-out)')
    if cfg.command == 'train' and cfg.warmup + cfg.cooldown >= cfg.num_epoch:
        n = cfg.warmup + cfg.cooldown + 1
        logger.warning(f'num_epoch ({cfg.num_epoch}) must exceed warmup + cooldown; using {n}')
        cfg.num_epoch = n
    if cfg.prefix is None:
        cfg.prefix = f"pelican-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cfg.run_dir = os.path.join(cfg.runs_dir, cfg.prefix)
    return cfg


def save_config(cfg, path):
    values = {k: v for k, v in sorted(vars(cfg).items()) if k not in _INTERNAL | {'run_dir'}}
    Path(path).write_text(yaml.safe_dump(values, sort_keys=False))
