"""
Export saved predictions to HDF5 for analysis outside PyTorch.

    python scripts/export_predictions.py runs/toptag-1            # every predictions/*.pt of a run
    python scripts/export_predictions.py runs/*/predictions/best_test.pt

Each <name>.pt (a dict with 'predict' and 'targets' tensors) becomes <name>.h5 next to it, one dataset per key.
"""
import argparse
import sys
from pathlib import Path

import h5py
import torch


def find_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            folder = p / 'predictions' if (p / 'predictions').is_dir() else p
            files += sorted(folder.glob('*.pt'))
        elif p.suffix == '.pt':
            files.append(p)
        else:
            print(f'skipping {p}: not a .pt file or folder', file=sys.stderr)
    return files


def export(path, force=False):
    out = path.with_suffix('.h5')
    if out.exists() and not force:
        print(f'{out} exists (use --force to overwrite)', file=sys.stderr)
        return None
    data = torch.load(path, map_location='cpu', weights_only=True)
    with h5py.File(out, 'w') as f:
        for key, value in data.items():
            if torch.is_tensor(value):
                f[key] = value.numpy()
            else:
                print(f'{path}: skipping non-tensor entry {key!r}', file=sys.stderr)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='*', default=['.'], help='run folders, predictions folders or .pt files (default: .)')
    parser.add_argument('--force', action='store_true', help='overwrite existing .h5 files')
    args = parser.parse_args(argv)
    files = find_files(args.paths)
    if not files:
        parser.error('no .pt prediction files found')
    for path in files:
        out = export(path, args.force)
        if out:
            print(f'{path} -> {out}')


if __name__ == '__main__':
    main()
