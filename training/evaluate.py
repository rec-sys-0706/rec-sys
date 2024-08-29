import numpy as np

def DCG(y_pred, y_true, k=10):
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
    ideal = DCG(y_true, y_true, k) # y_pred and y_true are paired lists.
    actual = DCG(y_pred, y_true, k)
    return np.nanmean(actual / ideal)

# def MRR(y_pred, y_true):
#     # TODO
#     rank = np.argsort(y_pred)[::-1]
#     y_true = np.take(y_true, rank)
#     rr_score = y_true / (np.arange(len(y_true)) + 1) # Assuming that y_true is one-hot vector.
#     return np.sum(rr_score) / np.sum(y_true)

def recall(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expect y_pred and y_ture have same shape but got {y_pred.shape} and {y_true.shape}.")

    # Calculate True Positives (TP) and False Negatives (FN)
    TP = np.sum(y_true * y_pred, axis=-1)
    FN = np.sum(y_true * (1 - y_pred), axis=-1)

    # Calculate Recall for each batch.
    recalls = TP / (TP + FN)

    # Calculate Mean Recall across all batches
    return np.mean(recalls)

def ROC_AUC(y_pred, y_true):
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

    # Compute true positive and false positive rates
    tps = np.cumsum(y_true, axis=-1)
    fps = np.cumsum(1 - y_true, axis=-1)

    if len(y_pred.shape) == 2:
        tpr = tps / tps[:, -1][:, None]
        fpr = fps / fps[:, -1][:, None]
    elif len(y_pred.shape) == 1:
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

    auc = np.trapz(tpr, fpr)
    return np.nanmean(auc)
