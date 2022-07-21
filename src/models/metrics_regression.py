import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from .lorentz_metric import normsq4, dot4

def metrics(predict, targets, prefix, logger=None):
    angle = AngleDeviation(predict, targets).item()
    massdelta = MassDeviation(predict, targets).item()
    loss_inv = loss_fn_inv(predict, targets)
    loss_m2 = loss_fn_m2(predict, targets)
    loss_3d = loss_fn_3d(predict, targets)
    loss_4d = loss_fn_4d(predict, targets)

    metrics = {'∆Ψ': angle, '∆m': massdelta, 'loss_inv': loss_inv, 'loss_m2': loss_m2, 'loss_3d': loss_3d, 'loss_4d': loss_4d}
    string = ' ∆Ψ: {:10.4f}, ∆m: {:10.4f}, loss_inv: {:10.4f}, loss_m2: {:10.4f}, loss_3d: {:10.4f}, loss_4d: {:10.4f}'.format(angle, massdelta, loss_inv, loss_m2, loss_3d, loss_4d)
    return metrics, string

def minibatch_metrics(predict, targets, entropy):
    angle = AngleDeviation(predict, targets).item()
    massdelta = MassDeviation(predict, targets).item()
    return [angle, massdelta]

def minibatch_metrics_string(metrics):
    string = ', ∆Ψ:{:> 9.4f}, ∆m:{:> 9.4f}'.format(*metrics)
    return string


# The loss function used for training is defined in the training script and passed to Trainer as an argument, here called self.loss_fn
# The Alternative losses defined below are used only for logging

def AngleDeviation(predict, targets):
    """
    Alternative metric for the log.
    """
    predict3 = predict[:,1:4]
    targets3 = targets[:,1:4]
    predict3 = predict3 / torch.norm(predict3, p=2, dim=-1, keepdim=True)
    targets3 = targets3 / torch.norm(targets3, p=2, dim=-1, keepdim=True)
    dots = torch.clamp(torch.einsum("ab,ab->a",predict3,targets3), -1 + 1e-7, 1 - 1e-7)
    angles = torch.acos(dots).unsqueeze(-1)
    if not torch.isnan(angles).any():
        return  angles.mean()
    else:
        return 0.0
    # return ((predict - targets).norm(dim=-1)/targets.norm(dim=-1)).mean()    # Euclidean distance relative error
    # return loss_fn_inv(predict,targets)
    # return ((predict[:,[1,2]] - targets[:,[1,2]]).norm(dim=-1)**2/targets[:,[1,2]].norm(dim=-1)**2).mean() # pT error relative norm
    # return (predict[:,[1,2,3]] - targets[:,[1,2,3]]).norm()/targets[:,[1,2,3]].norm()         # spatial momentum error relative norm

def MassDeviation(predict, targets):
    return ((normsq4(predict).abs().sqrt()-normsq4(targets).abs().sqrt())/normsq4(targets).abs().sqrt()).std()  # mass relative error
    # return loss_fn_m2(predict,targets)

def loss_fn_inv(predict, targets):
    return normsq4(predict - targets).abs().mean()

def loss_fn_m2(predict, targets):
    return (normsq4(predict) - normsq4(targets)).abs().mean()

def loss_fn_3d(predict, targets):
    return ((predict[:,[1,2,3]] - targets[:,[1,2,3]]).norm(dim=-1)).mean()

def loss_fn_4d(predict, targets):
    return (predict-targets).pow(2).sum(-1).mean()  
