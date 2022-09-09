import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from .lorentz_metric import normsq4, dot4

def metrics(predict, targets, loss_fn, prefix, logger=None):
    """
    This generates metrics reported at the end of each epoch and during validation/testing, as well as the logstring printed to the logger.
    """    
    loss = loss_fn(predict,targets).item()
    massdelta = MassSigma(predict, targets)
    massmean = MassMean(predict, targets))
    loss_m2 = loss_fn_m2(predict, targets)

    metrics = {'loss': loss, '∆m': massdelta, 'loss_m2': loss_m2}
    string = ' L: {:10.4f}, m/m: {:10.4f}, ∆m: {:10.4f}, loss_m2: {:10.4f}'.format(loss, massmean, massdelta, loss_m2)
    return metrics, string

def minibatch_metrics(predict, targets, loss):
    """
    This computes metrics for each minibatch (if verbose mode is used). The logstring is defined separately in minibatch_metrics_string.
    """    
    massmean = MassMean(predict, targets))
    massdelta = MassSigma(predict, targets)

    return [loss, massmean, massdelta]

def minibatch_metrics_string(metrics):
    string = '   L: {:12.4f}, m/m: {:9.4f}, ∆m: {:9.4f}'.format(*metrics)
    return string

def MassMean(predict, targets):
    """
    half of the 68% interquantile range over of relative deviation in mass
    """
    rel = (predict-normsq4(targets).abs().sqrt())/normsq4(targets).abs().sqrt()
    return rel.mean()  #  mean relative mass bias

def MassSigma(predict, targets):
    """
    half of the 68% interquantile range over of relative deviation in mass
    """
    rel = (predict-normsq4(targets).abs().sqrt())/normsq4(targets).abs().sqrt()
    return iqr(rel)  # mass relative error

def loss_fn_m2(predict, targets):
    return (predict - normsq4(targets).abs().sqrt()).abs().mean().item()

def iqr(x, rng=(0.16, 0.84)):
    rng = sorted(rng)
    return ((torch.quantile(x,rng[1])-torch.quantile(x,rng[0]))).item() / 2.