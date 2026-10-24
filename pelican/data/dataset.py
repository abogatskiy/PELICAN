"""HDF5 jet datasets. Each file holds equal-length arrays (one row per event), e.g. Pmu [E,N,4], Nobj, is_signal."""
import logging
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class JetDataset(Dataset):
    """
    One HDF5 file. `num_pts` < 0 uses every event. With `randomize_subset`, a random subset is chosen; with
    `balance` (binary targets), events are interleaved signal/background so unshuffled batches are balanced.
    `in_memory` loads the selected events into RAM (in chunks, keeping only the first `nobj` particles of
    per-particle arrays); otherwise items are read from disk on demand.
    """
    CHUNK = 100_000

    def __init__(self, filename, num_pts=-1, randomize_subset=True, balance=False, target='is_signal', in_memory=True, nobj=None):
        self.filename, self.in_memory = filename, in_memory
        self._file = None
        with h5py.File(filename, 'r') as f:
            keys = list(f.keys())
            length = len(f[keys[0]])
            self.keys = [k for k in keys if len(f[k]) == length]
            if 0 <= num_pts <= length:
                self.num_pts = num_pts
            else:
                if num_pts > length:
                    logger.warning(f'Requested {num_pts} events but {filename} has only {length}')
                self.num_pts = length

            self.perm = None
            if randomize_subset and (balance or self.num_pts < length):
                if balance:
                    if target not in f:
                        raise KeyError(f"Cannot balance classes: target '{target}' not found in {filename}")
                    labels = np.asarray(f[target][:])
                    if labels.ndim > 1:
                        labels = labels.argmax(-1)
                    sig, bkg = np.where(labels == 1)[0], np.where(labels == 0)[0]
                    sig, bkg = sig[torch.randperm(len(sig)).numpy()], bkg[torch.randperm(len(bkg)).numpy()]
                    n = min(len(sig), len(bkg))
                    idx = np.empty(len(sig) + len(bkg), dtype=np.int64)
                    idx[0:2 * n:2], idx[1:2 * n:2] = sig[:n], bkg[:n]
                    idx[2 * n:] = np.concatenate([sig[n:], bkg[n:]])
                    self.perm = idx[:self.num_pts]
                else:
                    self.perm = torch.randperm(length)[:self.num_pts].numpy()

            if in_memory:
                n_slots = f['Pmu'].shape[1] if 'Pmu' in f else None
                trim = lambda k, a: a[:, :nobj] if nobj and n_slots and a.ndim > 1 and a.shape[1] == n_slots else a
                if self.perm is None:
                    rows = np.arange(self.num_pts)
                else:
                    # read the needed rows in sorted order, then map item i -> its position
                    order = np.argsort(self.perm)
                    rows = self.perm[order]
                    self.perm = np.empty_like(order)
                    self.perm[order] = np.arange(len(order))
                self.data = {k: torch.from_numpy(self._read_rows(f[k], rows, lambda a: trim(k, a))) for k in self.keys}
            elif self.num_pts < length:
                logger.warning(f'Using {self.num_pts} of {length} events from {filename} without loading into RAM; reads may be slow')

    @classmethod
    def _read_rows(cls, dset, rows, transform):
        """dset[rows] for sorted rows, reading contiguous chunks so memory stays bounded."""
        out = []
        for start in range(0, int(rows[-1]) + 1 if len(rows) else 0, cls.CHUNK):
            sel = rows[(rows >= start) & (rows < start + cls.CHUNK)]
            if len(sel):
                out.append(transform(np.asarray(dset[start:min(start + cls.CHUNK, int(sel[-1]) + 1)]))[sel - start])
        return np.concatenate(out) if out else transform(np.asarray(dset[:0]))

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        if self.in_memory:
            return {k: v[idx] for k, v in self.data.items()}
        if self._file is None:                       # opened lazily, once per dataloader worker
            self._file = h5py.File(self.filename, 'r')
        return {k: torch.as_tensor(np.asarray(self._file[k][idx])) for k in self.keys}
