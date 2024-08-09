import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

def DCG(y_pred, y_true, k=10):
    """
        y_pred and y_true are paired lists.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expect y_pred and y_ture have same shape but got {y_pred.shape} and {y_true.shape}.")

    rank = np.argsort(-y_pred, axis=-1) # descending
    if len(y_pred.shape) == 2:
        y_true = y_true[np.arange(y_true.shape[0])[:, None], rank]
    elif len(y_pred.shape) == 1:
        y_true = np.take(y_true, rank)
    else:
        raise ValueError(f"Expect y_pred be 1-D or 2-D tensor.")

    gains = 2**y_true -1
    discounts = np.log2(np.arange(y_pred.shape[-1]) + 2)

    return np.sum(gains / discounts, axis=-1, keepdims=True)

def nDCG(y_pred, y_true, k=10):
    ideal = DCG(y_true, y_true, k)
    actual = DCG(y_pred, y_true, k)
    return np.nanmean(actual / ideal)

def MRR(y_pred, y_true):
    # TODO
    rank = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, rank)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def ROC_AUC(y_pred, y_true):
    # TODO too slow.
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expect y_pred and y_ture have same shape but got {y_pred.shape} and {y_true.shape}.")
    
    roc_auc_scores = np.array([roc_auc_score(y_true[i], y_pred[i]) for i in range(y_pred.shape[0])])

    return np.nanmean(roc_auc_scores)