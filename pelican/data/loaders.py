"""Find HDF5 files for each split and build (optionally distributed) dataloaders."""
import functools
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import JetDataset
from .collate import collate_fn

logger = logging.getLogger(__name__)
SPLITS = ('train', 'valid', 'test')


def find_files(datadir, splits=SPLITS, testfile=''):
    """HDF5 files in datadir are assigned to a split if the split name appears in the file name."""
    files = sorted(glob.glob(os.path.join(datadir, '*.h5')))
    found = {s: [f for f in files if s in os.path.basename(f)] for s in splits}
    if testfile and 'test' in splits:
        found['test'] = [testfile]
    for s in splits:
        if not found[s]:
            raise FileNotFoundError(f"No HDF5 file for split '{s}' in {datadir} (file names must contain '{s}')")
    return found


def _split_counts(total, nfiles):
    if total < 0:
        return [-1] * nfiles
    per = [total // nfiles] * nfiles
    per[-1] = total - sum(per[:-1])
    return per


def build_datasets(cfg, splits=SPLITS):
    files = find_files(cfg.datadir, splits, cfg.testfile)
    requested = {'train': cfg.num_train, 'valid': cfg.num_valid, 'test': cfg.num_test}
    balance = cfg.task == 'classification' and cfg.num_classes == 2 and cfg.balance
    datasets = {}
    for split in splits:
        counts = _split_counts(requested[split], len(files[split]))
        parts = [JetDataset(f, num_pts=n, randomize_subset=(split != 'test'), balance=balance and split != 'test',
                            target=cfg.target, in_memory=cfg.ram_dataset, nobj=cfg.nobj)
                 for f, n in zip(files[split], counts) if n != 0]
        datasets[split] = ConcatDataset(parts)
        logger.info(f'{split}: {len(datasets[split])} events from {len(parts)} file(s)')
    return datasets


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(cfg, splits=SPLITS, distributed=False):
    if cfg.fix_data:  # the same random subset of events across runs
        torch.manual_seed(165937750084982)
    datasets = build_datasets(cfg, splits)
    collate = functools.partial(collate_fn, scale=cfg.scale, nobj=cfg.nobj, dataset=cfg.dataset)
    loaders = {}
    for split, ds in datasets.items():
        sampler = None
        if distributed and (split != 'test' or cfg.distribute_eval):
            sampler = DistributedSampler(ds, shuffle=cfg.shuffle and split == 'train')
        loaders[split] = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                                    shuffle=cfg.shuffle and split == 'train' and sampler is None,
                                    num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                                    worker_init_fn=_seed_worker, collate_fn=collate)
    return loaders
