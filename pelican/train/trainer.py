"""
Training and evaluation loop. A run folder holds:
  config.yaml          resolved configuration           train.log      log of the run
  checkpoint.pt        last epoch (for --load)           best.pt        lowest validation loss
  metrics.csv          per-epoch train/valid metrics     test_metrics.json   best/final test metrics
  predictions/         <which>_<split>.pt (predictions + targets) and ROC curves as CSV
"""
import csv
import itertools
import json
import logging
import math
import os
import time
from datetime import datetime

import numpy as np
import torch

from . import distributed as D
from .optim import make_optimizer, make_scheduler

logger = logging.getLogger('pelican')


def _to(batch, device, dtype):
    """
    Move a batch to the device; floating tensors become the model dtype, except Pmu, which stays float64 where the
    device supports it so the canonical frame is computed from full-precision inputs. MPS has no float64, so there
    the host copy of Pmu is kept as '_Pmu_host' and the model computes the frame on the CPU.
    """
    out = {}
    if device.type == 'mps' and 'Pmu' in batch:
        out['_Pmu_host'] = batch['Pmu'].double()
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_floating_point():
            keep64 = k == 'Pmu' and device.type != 'mps'
            v = v.to(device, dtype=v.dtype if keep64 else dtype, non_blocking=True)
        elif torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
        out[k] = v
    return out


class Trainer:
    def __init__(self, cfg, model, task, loaders, device, ui, trial=None):
        self.cfg, self.task, self.loaders, self.device, self.ui, self.trial = cfg, task, loaders, device, ui, trial
        self.model = model
        self.net = model.module if hasattr(model, 'module') else model       # unwrapped (DDP)
        self.run_dir = cfg.run_dir
        self.main = D.is_main()
        self.optimizer = self.scheduler = None
        self.epoch, self.best_epoch, self.best_metrics = 0, 0, None
        self.writer = None
        if cfg.summarize and self.main:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(self.run_dir, 'tensorboard'))

    # ------------------------------------------------------------------ checkpoints
    def path(self, name):
        return os.path.join(self.run_dir, name)

    def _state(self):
        return {'model_state': self.net.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(), 'epoch': self.epoch, 'best_epoch': self.best_epoch,
                'best_metrics': self.best_metrics, 'config': {k: v for k, v in vars(self.cfg).items() if k != 'password'}}

    def save(self, name):
        if self.main:
            torch.save(self._state(), self.path(name))

    def resume(self):
        ckpt = torch.load(self.path('checkpoint.pt'), map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.epoch, self.best_epoch, self.best_metrics = ckpt['epoch'], ckpt['best_epoch'], ckpt['best_metrics']
        logger.info(f'Resumed from {self.path("checkpoint.pt")} after epoch {self.epoch} (best epoch {self.best_epoch})')

    # ------------------------------------------------------------------ training
    def fit(self):
        cfg = self.cfg
        steps = len(self.loaders['train'])
        self.optimizer = make_optimizer(cfg, self.net)
        self.scheduler = make_scheduler(cfg, self.optimizer, steps)
        if cfg.load and os.path.exists(self.path('checkpoint.pt')):
            self.resume()
        elif cfg.load:
            logger.warning('--load given but no checkpoint found; starting from scratch')

        for epoch in range(self.epoch + 1, cfg.num_epoch + 1):
            self.epoch = epoch
            t0 = time.time()
            sampler = getattr(self.loaders['train'], 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            train_metrics = self.train_epoch(epoch)
            valid_metrics = None
            predict, targets = self.predict('valid', distributed=True)
            if self.main:
                valid_metrics, _ = self.task.metrics(predict, targets)
                best = self.best_metrics is None or valid_metrics[self.task.best_metric] < self.best_metrics[self.task.best_metric]
                if best:
                    self.best_epoch, self.best_metrics = epoch, valid_metrics
                self.save('checkpoint.pt')
                if best:
                    self.save('best.pt')
                self._log_epoch(epoch, train_metrics, valid_metrics, time.time() - t0, best)
            D.barrier()
            if self.trial is not None:
                self._report_trial(epoch, valid_metrics)
        if self.writer:
            self.writer.close()
        return self.best_epoch, self.best_metrics

    def train_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()
        loader = self.loaders['train']
        preds, targs = [], []
        ema, skipped = None, 0
        with self.ui.epoch(epoch, cfg.num_epoch, len(loader)) as progress:
            for step, batch in enumerate(loader, 1):
                batch = _to(batch, self.device, self.net.dtype)
                targets = self.task.targets(batch)
                out = self.model(batch)
                loss = self.task.loss(out['predict'], targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.clip_grad)
                    finite = bool(torch.isfinite(gnorm)) and math.isfinite(loss.item())
                else:
                    finite = math.isfinite(loss.item())
                if finite:
                    self.optimizer.step()
                else:
                    skipped += 1
                self.scheduler.step()

                preds.append(D.all_gather(out['predict'].detach()).cpu())
                targs.append(D.all_gather(targets.detach()).cpu())
                if finite:
                    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
                stats = f'loss {ema:.4f}  lr {self.scheduler.get_last_lr()[0]:.2e}' if ema is not None else ''
                progress.update(step, stats)
                if cfg.save_every and step % cfg.save_every == 0:
                    self.save('checkpoint.pt')
        if skipped:
            logger.warning(f'epoch {epoch}: skipped {skipped} steps with non-finite loss or gradients')
        if not self.main:
            return None
        metrics, _ = self.task.metrics(torch.cat(preds), torch.cat(targs))
        return metrics

    def _log_epoch(self, epoch, train, valid, seconds, best):
        self.ui.epoch_summary(epoch, self.cfg.num_epoch, train, valid, self.scheduler.get_last_lr()[0], seconds, best, self.task.summary)
        path = self.path('metrics.csv')
        row = {'epoch': epoch, 'lr': self.scheduler.get_last_lr()[0], 'seconds': round(seconds, 1)}
        row.update({f'train_{k}': v for k, v in train.items()})
        row.update({f'valid_{k}': v for k, v in valid.items()})
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        if self.writer:
            for split, m in (('train', train), ('valid', valid)):
                for k, v in m.items():
                    self.writer.add_scalar(f'{split}/{k}', v, epoch)

    def _report_trial(self, epoch, valid_metrics):
        import optuna
        value = D.broadcast_object(valid_metrics[self.cfg.metric] if self.main else None)
        self.trial.report(value, epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def predict(self, split, distributed=True):
        """Predictions and targets on a split; gathered on rank 0 (others return None)."""
        if not distributed and not self.main:
            return None, None
        gather = D.all_gather if distributed else (lambda t: t)
        self.model.eval()
        model = self.model if distributed else self.net
        preds, targs = [], []
        for batch in self.loaders[split]:
            batch = _to(batch, self.device, self.net.dtype)
            preds.append(gather(model(batch)['predict']).cpu())
            targs.append(gather(self.task.targets(batch)).cpu())
        if not self.main:
            return None, None
        return torch.cat(preds), torch.cat(targs)

    def evaluate(self, splits=('test',), which=('best', 'final')):
        """Evaluate the best and final checkpoints. Writes test_metrics.json, predictions and a cross-run summary."""
        D.barrier()
        distributed = self.cfg.distribute_eval and D.world_size() > 1
        files = {'best': 'best.pt', 'final': 'checkpoint.pt'}
        results, cache = {}, {}
        for w in which:
            if not os.path.exists(self.path(files[w])):
                logger.warning(f'No {files[w]} in {self.run_dir}; skipping the {w} model')
                continue
            ckpt = torch.load(self.path(files[w]), map_location=self.device, weights_only=False)
            epoch = ckpt['epoch']
            for split in splits:
                if (epoch, split) in cache:                     # best == final: reuse
                    results[(w, split)] = dict(cache[(epoch, split)])
                    continue
                self.net.load_state_dict(ckpt['model_state'])
                predict, targets = self.predict(split, distributed)
                if not self.main:
                    continue
                metrics, artifacts = self.task.metrics(predict, targets)
                metrics['epoch'] = epoch
                results[(w, split)] = cache[(epoch, split)] = metrics
                if self.cfg.save_predictions:
                    self._save_predictions(w, split, predict, targets, artifacts)
        if self.main and results:
            with open(self.path('test_metrics.json'), 'w') as f:
                json.dump({f'{w}_{s}': m for (w, s), m in results.items()}, f, indent=2, default=float)
            self._append_summary(results)
            self.ui.results(results, ['epoch', 'loss', 'acc', 'auc', 'br30', 'br50', 'dPsi', 'dR', 'dpT', 'dm'])
        D.barrier()
        return results

    def _save_predictions(self, which, split, predict, targets, artifacts):
        out = os.path.join(self.run_dir, 'predictions')
        os.makedirs(out, exist_ok=True)
        torch.save({'predict': predict, 'targets': targets}, os.path.join(out, f'{which}_{split}.pt'))
        if 'roc' in artifacts:
            np.savetxt(os.path.join(out, f'{which}_{split}_roc.csv'), artifacts['roc'], delimiter=',', header='fpr,tpr,threshold')
        if 'roc_per_class' in artifacts:   # three columns (fpr, tpr, threshold) per class
            cols = list(itertools.zip_longest(*artifacts['roc_per_class'], fillvalue=0.))
            np.savetxt(os.path.join(out, f'{which}_{split}_roc.csv'), np.array(cols), delimiter=',')

    def _append_summary(self, results):
        """Runs whose prefixes differ only after a dash (seed-1, seed-2, ...) share runs/<base>.summary.csv."""
        base = self.cfg.prefix.split('-')[0]
        path = os.path.join(self.cfg.runs_dir, f'{base}.summary.csv')
        rows = [{'prefix': self.cfg.prefix, 'checkpoint': w, 'split': s, 'timestamp': datetime.now().isoformat(timespec='seconds'),
                 **{k: v for k, v in m.items() if np.isscalar(v)}} for (w, s), m in results.items()]
        fields = list(rows[0])
        new = not os.path.exists(path)
        if not new:
            with open(path) as f:
                header = next(csv.reader(f), [])
            if header != fields:
                fields = header or fields
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerows(rows)
