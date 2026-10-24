"""Training objectives: loss, evaluation metrics, a compact summary for the console, and saved artifacts."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from . import regression_metrics as rm
from . import multiclass_metrics as mm


def _rejection(fpr, tpr, at):
    """Background rejection 1/eB at the ROC point closest to signal efficiency `at` (eB > 0)."""
    idx = np.argmin(np.abs(tpr - at))
    if fpr[idx] <= 0:
        pos = np.where(fpr > 0)[0]
        if len(pos) == 0:
            return 0.0, 1.0
        idx = pos[0]
    return float(1.0 / fpr[idx]), float(tpr[idx])


class ClassificationTask:
    name = 'classification'
    best_metric, best_mode = 'loss', 'min'

    def __init__(self, num_classes, target):
        self.num_classes, self.target = num_classes, target

    def targets(self, batch):
        t = batch[self.target]
        return (t.argmax(-1) if t.dim() > 1 else t).long()

    def loss(self, predict, targets):
        return F.cross_entropy(predict, targets)

    def metrics(self, predict, targets):
        """predict: logits [E,C], targets: class indices [E]. Returns (scalar metrics, artifacts)."""
        predict, targets = predict.double(), targets.long()
        probs = predict.softmax(-1)
        out = {'loss': F.cross_entropy(predict, targets).item(),
               'acc': (probs.argmax(-1) == targets).double().mean().item()}
        artifacts = {}
        if self.num_classes == 2:
            y, s = targets.numpy(), probs[:, 1].numpy()
            if 0 < y.sum() < len(y):
                fpr, tpr, thr = roc_curve(y, s)
                out['auc'] = float(roc_auc_score(y, s))
                out['br30'], out['eS30'] = _rejection(fpr, tpr, 0.3)
                out['br50'], out['eS50'] = _rejection(fpr, tpr, 0.5)
                artifacts['roc'] = np.stack([fpr, tpr, thr], 1)
        else:
            onehot = F.one_hot(targets, self.num_classes).double()
            out['auc'] = float(mm.AUCScoreMacro(probs, onehot))
            curves, eB30, _, eB50, *_ = mm.ROC(probs, onehot)
            aucs = mm.AUCScore(probs, onehot)
            for c in range(self.num_classes):
                out[f'auc_{c}'] = float(aucs[c])
                out[f'br30_{c}'] = float(1 / eB30[c]) if eB30[c] > 0 else 0.0
                out[f'br50_{c}'] = float(1 / eB50[c]) if eB50[c] > 0 else 0.0
            artifacts['roc_per_class'] = curves
        return out, artifacts

    @staticmethod
    def summary(m):
        parts = [f"loss {m['loss']:.4f}", f"acc {m['acc']:.4f}"]
        if 'auc' in m:
            parts.append(f"auc {m['auc']:.4f}")
        if 'br30' in m:
            parts.append(f"br30 {m['br30']:.0f}")
        return '  '.join(parts)


# name in --loss -> loss function on predicted/target 4-vectors
REGRESSION_LOSSES = {'m': rm.loss_fn_m, 'm2': rm.loss_fn_m2, '3d': rm.loss_fn_3d, '4d': rm.loss_fn_4d, 'E': rm.loss_fn_E,
                     'psi': rm.loss_fn_psi, 'pT': rm.loss_fn_pT, 'dR': rm.loss_fn_dR, 'inv': rm.loss_fn_inv,
                     'col': rm.loss_fn_col, 'col3': rm.loss_fn_col3}


def parse_loss(spec):
    terms = []
    for item in spec.split(','):
        name, _, weight = item.strip().partition(':')
        if name not in REGRESSION_LOSSES:
            raise ValueError(f"Unknown regression loss '{name}' (choose from {', '.join(REGRESSION_LOSSES)})")
        terms.append((name, float(weight or 1.0)))
    return terms


class RegressionTask:
    name = 'regression'
    best_metric, best_mode = 'loss', 'min'

    def __init__(self, loss_spec, target, scale=1.0):
        self.terms = parse_loss(loss_spec)
        self.target = target

    def targets(self, batch):
        return batch[self.target].to(batch['Pmu'].dtype)

    def loss(self, predict, targets):
        targets = targets.to(predict.dtype)
        return sum(w * REGRESSION_LOSSES[n](predict, targets) for n, w in self.terms)

    def metrics(self, predict, targets):
        predict, targets = predict.double(), targets.double()
        scalar = lambda t: float(t.mean()) if t.numel() > 1 else float(t)
        out = {'loss': float(self.loss(predict, targets)),
               'dPsi': scalar(rm.AngleDeviation(predict, targets)),
               'dR': scalar(rm.dRSigma(predict, targets)),
               'dpT': scalar(rm.pTSigma(predict, targets)),
               'dm': scalar(rm.MassSigma(predict, targets))}
        for n in REGRESSION_LOSSES:
            out[f'loss_{n}'] = float(REGRESSION_LOSSES[n](predict, targets))
        return out, {}

    @staticmethod
    def summary(m):
        return f"loss {m['loss']:.4f}  ∆Ψ {m['dPsi']:.4f}  ∆R {m['dR']:.4f}  ∆pT {m['dpT']:.4f}  ∆m {m['dm']:.4f}"


def make_task(cfg):
    if cfg.task == 'classification':
        return ClassificationTask(cfg.num_classes, cfg.target)
    return RegressionTask(cfg.loss, cfg.target, cfg.scale)
