"""
Convert checkpoints of the pre-package scripts (train_pelican_classifier.py / train_pelican_cov.py) into a run folder.

    python scripts/convert_checkpoint.py model/classifier.pt                 # -> runs/classifier/
    python scripts/convert_checkpoint.py model/old_best.pt --prefix new-name --runs-dir runs

Given model/<name>.pt and/or model/<name>_best.pt, writes <runs-dir>/<prefix>/config.yaml (the old arguments translated
to the package's options), checkpoint.pt and best.pt, with the parameters renamed for the package. The converted run
can then be evaluated with `pelican eval runs/<prefix>` (e.g. --datadir or --testfile to point at the data), but not
resumed with --load: the optimizer and learning-rate schedule changed.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

from pelican.cli import build_model
from pelican.config import build_parser, save_config, validate
from pelican.legacy import convert_args, convert_state_dict, load_checkpoint


def checkpoint_files(path):
    path = Path(path)
    stem = path.stem[:-len('_best')] if path.stem.endswith('_best') else path.stem
    final, best = path.with_name(f'{stem}.pt'), path.with_name(f'{stem}_best.pt')
    return stem, (final if final.exists() else None), (best if best.exists() else None)


def convert(path, prefix=None, runs_dir='runs', force=False):
    stem, final, best = checkpoint_files(path)
    if final is None and best is None:
        raise SystemExit(f'no checkpoint found for {path}')
    prefix = prefix or stem
    run_dir = os.path.join(runs_dir, prefix)
    if os.path.exists(os.path.join(run_dir, 'config.yaml')) and not force:
        raise SystemExit(f'{run_dir} already exists (use --force to overwrite)')
    ckpts = {name: load_checkpoint(f) for name, f in (('checkpoint.pt', final), ('best.pt', best)) if f}
    first = next(iter(ckpts.values()))
    values, ignored = convert_args(first['args'], first['model_state'])
    cfg = build_parser().parse_args(['train'])
    for key, value in values.items():
        setattr(cfg, key, value)
    cfg.prefix, cfg.runs_dir, cfg.command = prefix, runs_dir, 'train'
    cfg = validate(cfg)
    states = {name: convert_state_dict(ckpt['model_state']) for name, ckpt in ckpts.items()}
    try:
        model = build_model(cfg)
        for state in states.values():
            model.load_state_dict(state, strict=True)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f'{path}: cannot be converted (probably from an earlier PELICAN architecture): {str(e).splitlines()[0]}')
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, 'config.yaml'))
    for name, ckpt in ckpts.items():
        torch.save({'model_state': states[name], 'epoch': ckpt.get('epoch', 0),
                    'best_epoch': ckpt.get('best_epoch', ckpt.get('epoch', 0)), 'best_metrics': ckpt.get('best_metrics'),
                    'config': dict(vars(cfg))}, os.path.join(run_dir, name))
    print(f"{path} -> {run_dir} ({', '.join(ckpts)}); ignored old options: {', '.join(sorted(ignored)) or 'none'}")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', help='old checkpoint files (model/<name>.pt or model/<name>_best.pt)')
    parser.add_argument('--prefix', default=None, help='run name (default: the checkpoint name; only with one checkpoint)')
    parser.add_argument('--runs-dir', default='runs', help='parent directory of run folders (default: runs)')
    parser.add_argument('--force', action='store_true', help='overwrite an existing run folder')
    args = parser.parse_args(argv)
    if args.prefix and len(args.checkpoints) > 1:
        parser.error('--prefix needs a single checkpoint')
    for path in args.checkpoints:
        convert(path, args.prefix, args.runs_dir, args.force)


if __name__ == '__main__':
    sys.exit(main())
