import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import itertools

def metrics(predict, targets, loss_fn, prefix, logger=None):
    loss = torch.nn.CrossEntropyLoss()(predict, targets).item()
    predict = predict.softmax(dim=1)
    num_classes = predict.shape[-1]
    accuracy = Accuracy(predict, targets).item()
    auc_ovr = AUCScore(predict, targets)
    auc_macro = AUCScoreMacro(predict, targets)
    roc, eB03s, eS03s, eB05s, eS05s, tpr10s, fpr10s, tpr1s, fpr1s = ROC(predict, targets)
    conf_matrix = confusion_matrix(targets.argmax(dim=1), predict.argmax(dim=1), normalize='true')
    report = classification_report(targets.argmax(dim=1), predict.argmax(dim=1), target_names=[str(i) for i in range(num_classes)])
    metrics = {'loss': loss, 'accuracy': accuracy}
    metrics.update({f'AUC_macro': auc_macro})
    metrics.update({f'AUC | {c}': auc for c, auc in enumerate(auc_ovr)})
    metrics.update({f'eS at eB=0.1 | {c}': tpr for c, tpr in enumerate(tpr10s)})
    metrics.update({f'eB at eB=0.1 | {c}': fpr for c, fpr in enumerate(fpr10s)})
    metrics.update({f'eS at eB=0.01 | {c}': tpr for c, tpr in enumerate(tpr1s)})
    metrics.update({f'eB at eB=0.01 | {c}': fpr for c, fpr in enumerate(fpr1s)})
    metrics.update({f'1/eB at eS=0.3 | {c}': 1/eB if eB>0 else 0 for c, eB in enumerate(eB03s)})
    metrics.update({f'eS at eS=0.3 | {c}': eS for c, eS in enumerate(eS03s)})
    metrics.update({f'1/eB at eS=0.5 | {c}': 1/eB if eB>0 else 0 for c, eB in enumerate(eB05s)})
    metrics.update({f'eS at eS=0.5 | {c}': eS for c, eS in enumerate(eS05s)})
    metrics.update({'conf': conf_matrix, 'report': report})
    string = ' L: {:10.4f}, ACC: {:10.4f}, AUC: {}, AUCs: {},    eS: {} @ {:>4.2f},    eS: {} @ {:>4.2f},    1/eB: {} @ {:>4.2f},    1/eB: {} @ {:>4.2f},\nconf:\n{},\nreport:\n{}'.format(loss, accuracy, auc_macro, auc_ovr, tpr10s, 0.1, tpr1s, 0.01, [1/eB03 if eB03>0 else 0 for eB03 in eB03s], 0.3,  [1/eB05 if eB05>0 else 0 for eB05 in eB05s], 0.5, conf_matrix, report)
    np.savetxt(prefix+'_ROC.csv', np.array(list(itertools.zip_longest(*roc,fillvalue=0.))), delimiter=',')
    # if logger:
    #     logger.info('ROC saved to file ' + prefix+'_ROC.csv' + '\n')
    return metrics, string

def minibatch_metrics(predict, targets, loss):
    predict = predict.softmax(dim=1)
    accuracy = Accuracy(predict, targets).item()
    auc_score = AUCScoreMacro(predict, targets)
    return [loss, accuracy, auc_score]

def minibatch_metrics_string(metrics):
    string = ', L:{:> 9.4f}, ACC:{:> 9.4f}, AUC:{:> 9.4f}'.format(*metrics)
    return string



def Entropy(predict, targets):
    return torch.nn.CrossEntropyLoss()(predict, targets.long())      # Cross Entropy Loss (positive number). The closer to 0 the better.

def Accuracy(predict, targets):
    return (predict.argmax(dim=1) == targets.long().argmax(dim=1)).float().mean()  # right now this is accuracy of classification

# AUC score for logging
def AUCScore(predict, targets):
    num_classes = targets.shape[-1]
    scores=[]
    for c in range(num_classes):
        if torch.equal(targets[...,c], torch.ones_like(targets[...,c])) or torch.equal(targets[...,c], torch.zeros_like(targets[...,c])):
            scores.append(0)
        else:
            scores.append(roc_auc_score(targets[...,c], predict[...,c], multi_class='ovr'))        # Area Under Curve score (between 0 and 1). The closer to 1 the better.
    return np.array(scores)

def AUCScoreMacro(predict, targets):
    return roc_auc_score(targets, predict, multi_class='ovo', average='macro')


def ROC(predict, targets):
    num_classes = targets.shape[-1]
    curves = ()
    eB03s, eB05s, eS03s, eS05s   = [], [], [], []
    tpr10s, fpr10s, tpr1s, fpr1s = [], [], [], []
    for c in range(num_classes):
        if torch.equal(targets[...,c], torch.ones_like(targets[...,c])) or torch.equal(targets[...,c], torch.zeros_like(targets[...,c])):
            curves = curves + (np.array([]),np.array([]),np.array([]))
            [x.append(0.) for x in [eB03s, eB05s, eS03s, eS05s, tpr10s, fpr10s, tpr1s, fpr1s]]
        else:
            curve = roc_curve(targets[...,c], predict[..., c])
            eB03, eS03 = BR(curve, at_eS=0.3)
            eB05, eS05 = BR(curve, at_eS=0.5)
            curves = curves + curve
            eB03s.append(eB03)
            eB05s.append(eB05)
            eS03s.append(eS03)
            eS05s.append(eS05)
            fpr10, tpr10 = TPRatFPR(curve, at_eB=0.1)
            fpr1, tpr1 = TPRatFPR(curve, at_eB=0.01)
            fpr10s.append(fpr10)
            tpr10s.append(tpr10)
            fpr1s.append(fpr1)
            tpr1s.append(tpr1)
    return curves, np.array(eB03s), np.array(eS03s), np.array(eB05s), np.array(eS05s), np.array(tpr10s), np.array(fpr10s), np.array(tpr1s), np.array(fpr1s)

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