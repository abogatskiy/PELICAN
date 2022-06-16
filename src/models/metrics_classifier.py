import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def metrics(predict, targets, prefix, logger=None):
    entropy = Entropy(predict, targets).item()
    accuracy = Accuracy(predict, targets).item()
    auc_score = AUCScore(predict, targets)
    roc, eB, eS = ROC(predict, targets)
    conf_matrix = confusion_matrix(targets, predict.argmax(dim=1)) / targets.shape[0]
    metrics = {'loss': entropy, 'accuracy': accuracy, 'AUC': auc_score, 'BgRejection': 1/eB if eB>0 else 0, 'atSignEfficiency': eS, 'FP_rate': conf_matrix[0,1], 'FN_rate': conf_matrix[1,0]}
    string = ' L: {:10.4f}, ACC: {:10.4f}, AUC: {:10.4f},    BR: {:10.1f} @ {:>4.4f},   FP: {:10.4f}, FN: {:10.4f}'.format(entropy, accuracy, auc_score, 1/eB if eB>0 else 0, eS, conf_matrix[0,1], conf_matrix[1,0])
    np.savetxt(prefix+'_ROC.csv', ROC(predict, targets)[0], delimiter=',')
    # if logger:
    #     logger.info('ROC saved to file ' + prefix+'_ROC.csv' + '\n')
    return metrics, string

def minibatch_metrics(predict, targets, entropy):
    accuracy = Accuracy(predict, targets).item()
    auc_score = AUCScore(predict, targets)
    return [entropy, accuracy, auc_score]

def minibatch_metrics_string(metrics):
    string = ', L:{:> 9.4f}, ACC:{:> 9.4f}, AUC:{:> 9.4f}'.format(*metrics)
    return string


def Entropy(predict, targets):
    return torch.nn.CrossEntropyLoss()(predict, targets.long())      # Cross Entropy Loss (positive number). The closer to 0 the better.

def Accuracy(predict, targets):
    return (predict.argmax(dim=1) == targets.long()).float().mean()  # right now this is accuracy of classification

# AUC score for logging
def AUCScore(predict, targets):
    if torch.equal(targets, torch.ones_like(targets)) or torch.equal(targets, torch.zeros_like(targets)):
        return 0
    else:
        return roc_auc_score(targets, predict[:, 1])          # Area Under Curve score (between 0 and 1). The closer to 1 the better.

def ROC(predict, targets):
    if torch.equal(targets, torch.ones_like(targets)) or torch.equal(targets, torch.zeros_like(targets)):
        return None, 0., 0.
    else:
        curve = roc_curve(targets, predict[:, 1])
        idx = np.argmin(np.abs(curve[1]-0.3))
        if curve[0][idx]>0.: 
            eB, eS = curve[0][idx], curve[1][idx]
        else:
            idx = np.where(curve[0]>0)[0]
            if len(idx)>0:
                idx = idx[0]
                eB, eS = curve[0][idx], curve[1][idx]
            else:
                eB, eS = 1., 1.
        return curve, eB, eS
