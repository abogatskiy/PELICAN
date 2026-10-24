"""Per-class ROC metrics for multiclass classification (targets one-hot), carried over from PELICAN."""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def AUCScore(predict, targets):
    num_classes = targets.shape[-1]
    scores=[]
    for c in range(num_classes):
        if torch.equal(targets[...,c], torch.ones_like(targets[...,c])) or torch.equal(targets[...,c], torch.zeros_like(targets[...,c])):
            scores.append(0)
        else:
            # AUC for each class
            scores.append(roc_auc_score(targets[...,c], predict[...,c], multi_class='ovr'))
    return np.array(scores)

def AUCScoreMacro(predict, targets):
    # If any of the classes are not represented in the sample, macro AUC can't be computed
    if (targets.sum(0)==0).any():
        return 0
    return roc_auc_score(targets, predict, multi_class='ovo', average='macro')


def ROC(predict, targets, backg_class=-1):
    num_classes = targets.shape[-1]
    curves = ()
    eB30s, eB50s, eB995s, eS30s, eS50s, eS995s   = [], [], [], [], [], []
    tpr10s, fpr10s, tpr1s, fpr1s = [], [], [], []
    for c in range(num_classes):
        if torch.equal(targets[...,c], torch.ones_like(targets[...,c])) or torch.equal(targets[...,c], torch.zeros_like(targets[...,c])):
            curves = curves + (np.array([]),np.array([]),np.array([]))
            [x.append(0.) for x in [eB30s, eB50s, eS30s, eS50s, tpr10s, fpr10s, tpr1s, fpr1s]]
        else:
            if backg_class >= 0:
                eps = 1e-12
                scores = predict[..., c]/(eps + predict[..., c] + predict[..., backg_class])
                curve = roc_curve(targets[..., c], scores)
            else:
                curve = roc_curve(targets[...,c], predict[..., c])
            eB30, eS30   = BR(curve, at_eS=0.3)
            eB50, eS50   = BR(curve, at_eS=0.5)
            eB995, eS995 = BR(curve, at_eS=0.995)
            curves = curves + curve
            eB30s.append(eB30)
            eB50s.append(eB50)
            eB995s.append(eB995)
            eS30s.append(eS30)
            eS50s.append(eS50)
            eS995s.append(eS995)
            fpr10, tpr10 = TPRatFPR(curve, at_eB=0.1)
            fpr1, tpr1 = TPRatFPR(curve, at_eB=0.01)
            fpr10s.append(fpr10)
            tpr10s.append(tpr10)
            fpr1s.append(fpr1)
            tpr1s.append(tpr1)
    return curves, np.array(eB30s), np.array(eS30s), np.array(eB50s), np.array(eS50s), np.array(eB995s), np.array(eS995s), np.array(tpr10s), np.array(fpr10s), np.array(tpr1s), np.array(fpr1s)

def BR(curve, at_eS):
    # Given an ROC curve defined as (eB, eS) aka (FPR, TPR), return the point on it closest to the requested signal efficiency
    idx = np.argmin(np.abs(curve[1]-at_eS))
    if curve[0][idx]>0.: 
        eB, eS = curve[0][idx], curve[1][idx]
    else:
        idx = np.where(curve[0]>0)[0]
        if len(idx)>0:
            idx = idx[0]
            eB, eS = curve[0][idx], curve[1][idx]
        else:
            eB, eS = 1., 1.
    return eB, eS

def TPRatFPR(curve, at_eB):
    # Given an ROC curve defined as (eB, eS) aka (FPR, TPR), return the point on it closest to the requested FPR (eB)
    idx = np.argmin(np.abs(curve[0]-at_eB))
    if curve[1][idx]>0.: 
        eB, eS = curve[0][idx], curve[1][idx]
    else:
        idx = np.where(curve[1]>0)[0]
        if len(idx)>0:
            idx = idx[0]
            eB, eS = curve[0][idx], curve[1][idx]
        else:
            eB, eS = 1., 1.
    return eB, eS
