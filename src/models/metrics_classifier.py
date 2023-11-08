import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def metrics(predict, targets, loss_fn, prefix, logger=None):
    loss = loss_fn(predict, targets.long()).item()
    predict = predict.softmax(dim=1)
    accuracy = Accuracy(predict, targets).item()
    auc_score = AUCScore(predict, targets)
    roc, eB03, eS03, eB05, eS05 = ROC(predict, targets)
    conf_matrix = confusion_matrix(targets, predict.argmax(dim=1), normalize='true')
    metrics = {'loss': loss, 'accuracy': accuracy, 'AUC': auc_score, 'BgRejectionAt0.3': 1/eB03 if eB03>0 else 0, 'atSignEfficiency03': eS03, 'BgRejectionAt0.5': 1/eB05 if eB05>0 else 0, 'atSignEfficiency05': eS05, 'FP_rate': conf_matrix[0,1], 'FN_rate': conf_matrix[1,0]}
    string = ' L: {:10.4f}, ACC: {:10.4f}, AUC: {:10.4f},    BR: {:10.1f} @ {:>4.4f},    BR: {:10.1f} @ {:>4.4f},   FP: {:10.4f}, FN: {:10.4f}'.format(loss, accuracy, auc_score, 1/eB03 if eB03>0 else 0, eS03,  1/eB05 if eB05>0 else 0, eS05, conf_matrix[0,1], conf_matrix[1,0])
    np.savetxt(prefix+'_ROC.csv', roc, delimiter=',')
    # if logger:
    #     logger.info('ROC saved to file ' + prefix+'_ROC.csv' + '\n')
    return metrics, string

def minibatch_metrics(predict, targets, loss):
    predict = predict.softmax(dim=1)
    accuracy = Accuracy(predict, targets).item()
    auc_score = AUCScore(predict, targets)
    return [loss, accuracy, auc_score]

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
        return roc_auc_score(targets, predict[:, -1])          # Area Under Curve score (between 0 and 1). The closer to 1 the better.

def ROC(predict, targets):
    if torch.equal(targets, torch.ones_like(targets)) or torch.equal(targets, torch.zeros_like(targets)):
        return None, 0., 0., 0., 0.
    else:
        curve = roc_curve(targets, predict[:, -1])
        eB03, eS03 = BR(curve, at_eS=0.3)
        eB05, eS05 = BR(curve, at_eS=0.5)
        return curve, eB03, eS03, eB05, eS05

def BR(curve, at_eS):
    # Given an ROC curve defined as (eB, eS), return the point on it closest to the requested signal efficiency
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