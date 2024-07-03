import torch
from torch.utils.data import Dataset

import h5py
import numpy as np

import logging
logger = logging.getLogger(__name__)

class JetDataset(Dataset):
    """
    PyTorch dataset.
    """
    def __init__(self, filename, num_pts=-1, randomize_subset=True, balance=True, RAMdataset=False):

        self.filename = filename
        self.RAMdataset = RAMdataset

        with h5py.File(filename, mode='r') as f:
            len_data = len(f[next(iter(f))])
            if num_pts < 0:
                self.num_pts = len_data
            else:
                if num_pts > len_data:
                    logging.warn('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(filename['Nobj'])))
                    self.num_pts = len_data
                else:
                    self.num_pts = num_pts

            # Shuffle the dataset before initializing the dataloader
            # (mainly so that if num_pts<len_data we get a random subset and not just the first num_pts events)
            if randomize_subset and (balance or num_pts < len_data):
                # This can be used to balance batches in binary datasets:
                # The dataset is re-ordered so that the true labels have the form [0,1,0,1,0,1,...]
                # Assuming shuffle=False in the dataloader, this will guarantee balanced minibatches
                # TODO ideally this should be achieved by a custom sampler, not here
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

                    # Now interweave signal and background indices.
                    idxs = np.arange((signal_idxs.size + backgd_idxs.size),dtype=signal_idxs.dtype)
                    idxs[0:2*num_pairs:2] = signal_idxs[:num_pairs]
                    idxs[1:2*num_pairs:2] = backgd_idxs[:num_pairs]
                    idxs[2*num_pairs:] = np.concatenate([signal_idxs[num_pairs:], backgd_idxs[num_pairs:]])

                    self.perm = idxs[:self.num_pts]
                elif num_pts < len_data:
                    self.perm = torch.randperm(len_data)[:self.num_pts]
            else:
                self.perm = None

            if RAMdataset:
                logger.warn(f'Reading {self.num_pts} events from {filename} into RAM.')
                if self.perm is None:
                    self.data = {key: torch.from_numpy(val[:self.num_pts]) for key, val in f.items() if len(val)==len_data}
                else:
                    # this returns the permutation of range(num_pts) that unsorts sorted(self.perm), and subset=sorted(self.perm)
                    self.perm, subset = zip(*list(sorted(enumerate(self.perm), key=lambda x: x[1])))
                    # only load data[subset] into RAM 
                    self.data = {key: torch.from_numpy(val[:])[torch.tensor(subset)] for key, val in f.items()}
                    # this version will only ever read num_pts events into RAM, but will be slow because subset is not sequential
                    # self.data = {key: torch.from_numpy(val[list(subset)]) for key, val in f.items()}
            elif num_pts > 0 and num_pts < len_data:
                logger.warn(f'Chose {num_pts} event indices from {filename}. Batches will be read directly from disk (might be slow!).')

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if not self.RAMdataset:
            self.data = h5py.File(self.filename,'r')
        if self.perm is not None:
            idx = self.perm[idx]
        item = {key: val[idx] for key, val in self.data.items()}
        if not self.RAMdataset:
            item = {key: torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.tensor(val) for key, val in item.items()}
        return item
