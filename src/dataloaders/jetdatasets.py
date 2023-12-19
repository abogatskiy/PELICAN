import torch
from torch.utils.data import Dataset

import h5py
from itertools import islice
from math import inf
import numpy as np

import logging

class JetDataset(Dataset):
    """
    PyTorch dataset.
    """
    def __init__(self, filename, num_pts=-1, shuffle=True, balance=True):

        self.filename = filename
        self.data = filename

        with h5py.File(filename, mode='r') as f:
            if num_pts < 0:
                self.num_pts = len(f[next(iter(f))])
            else:
                if num_pts > len(f[next(iter(f))]):
                    logging.warn('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(filename['Nobj'])))
                    self.num_pts = len(f[next(iter(f))])
                else:
                    self.num_pts = num_pts

            if shuffle: # shuffle the order of data on initialization, w.r.t. the ordering in the input file(s)
                if balance:
                    # We want to shuffle things, but make sure that we keep our signal-to-background ratio
                    # We will assume there are only two possible values (0,1) of "is_signal". #TODO: We could consider generalizing/extending this, for multi-label classification problems.
                    signal_flags = f['is_signal'][:] # This will read the entire column into RAM, which is a potential limitation
                    signal_idxs = np.where(signal_flags==1)[0]
                    backgd_idxs = np.where(signal_flags==0)[0]

                    # We will randomly permute each list of indices using PyTorch, so that its RNG (which is global) is invoked.
                    signal_perm = torch.randperm(len(signal_idxs))
                    backgd_perm = torch.randperm(len(backgd_idxs))

                    signal_idxs = signal_idxs[signal_perm]
                    backgd_idxs = backgd_idxs[backgd_perm]
                    
                    num_pairs = min(signal_idxs.size, backgd_idxs.size)
                    # Now interleave signal and background indices, so we have a list of them all.
                    idxs = np.arange((signal_idxs.size + backgd_idxs.size),dtype=signal_idxs.dtype)

                    idxs[0:2*num_pairs:2] = signal_idxs[:num_pairs]
                    idxs[1:2*num_pairs:2] = backgd_idxs[:num_pairs]
                    idxs[2*num_pairs:] = np.concatenate([signal_idxs[num_pairs:], backgd_idxs[num_pairs:]])

                    self.perm = idxs[:self.num_pts]
                else:
                    self.perm = torch.randperm(len(filename['Nobj']))[:self.num_pts]
            else:
                self.perm = None


    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        self.data = h5py.File(self.filename,'r')
        if self.perm is not None:
            idx = self.perm[idx]
        item = {key: val[idx] for key, val in self.data.items()}
        item = {key: torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.tensor(val) for key, val in item.items()}
        return item
