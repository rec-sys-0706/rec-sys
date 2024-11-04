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

    gains = 2**y_true - 1
    discounts = np.log2(np.arange(y_pred.shape[-1]) + 2)

    return np.sum(gains / discounts, axis=-1, keepdims=True)

def nDCG(y_pred, y_true, k=10):
    ideal = DCG(y_true, y_true, k) # y_pred and y_true are paired lists.
    actual = DCG(y_pred, y_true, k)
    return np.nanmean(actual / (ideal + np.finfo(float).eps))

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
    y_pred = np.where(y_pred > 0.5, 1, 0)
    # Calculate True Positives (TP) and False Negatives (FN)
    TP = np.sum(y_true * y_pred, axis=-1)
    FN = np.sum(y_true * (1 - y_pred), axis=-1)

    # Calculate Recall for each batch.
    recalls = TP / (TP + FN + np.finfo(float).eps)

    # Calculate Mean Recall across all batches
    return np.mean(recalls)

def accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expect y_pred and y_ture have same shape but got {y_pred.shape} and {y_true.shape}.")
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return np.mean(y_pred == y_true)
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
        tpr = tps / (tps[:, -1][:, None] + np.finfo(float).eps)
        fpr = fps / (fps[:, -1][:, None] + np.finfo(float).eps)
    elif len(y_pred.shape) == 1:
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

    auc = np.trapz(tpr, fpr)
    return np.nanmean(auc)

def DCG_new(y_pred, y_true, k=10):
    # Ensure y_pred and y_true are lists of lists
    if len(y_pred) != len(y_true):
        raise ValueError(f"Expect y_pred and y_true to have the same number of rows, but got {len(y_pred)} and {len(y_true)}.")
    
    dcg_scores = []
    for pred, true in zip(y_pred, y_true):
        # Convert each row to numpy arrays
        pred = np.array(pred)
        true = np.array(true)

        # Sort true labels by the descending order of predictions
        rank = np.argsort(-pred)
        true = true[rank[:k]]

        # Calculate gains and discounts
        gains = 2 ** true - 1
        discounts = np.log2(np.arange(len(true)) + 2)

        # Calculate DCG for each row
        dcg_scores.append(np.sum(gains / discounts))
    
    return np.array(dcg_scores)

def nDCG_new(y_pred, y_true, k=10):
    # Calculate ideal DCG
    ideal_dcg_scores = DCG_new(y_true, y_true, k)
    actual_dcg_scores = DCG_new(y_pred, y_true, k)

    # Avoid division by zero
    return np.nanmean(actual_dcg_scores / (ideal_dcg_scores + np.finfo(float).eps))

def recall_new(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError(f"Expect y_pred and y_true to have the same number of rows, but got {len(y_pred)} and {len(y_true)}.")
    
    recalls = []
    for pred, true in zip(y_pred, y_true):
        # Binarize predictions based on threshold
        pred = np.array(pred) > 0.5
        true = np.array(true)
        
        TP = np.sum(true * pred)
        FN = np.sum(true * (1 - pred))
        
        recalls.append(TP / (TP + FN + np.finfo(float).eps))
    
    return np.mean(recalls)

def accuracy_new(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError(f"Expect y_pred and y_true to have the same number of rows, but got {len(y_pred)} and {len(y_true)}.")
    
    accuracies = []
    for pred, true in zip(y_pred, y_true):
        # Binarize predictions based on threshold
        pred = np.array(pred) > 0.5
        true = np.array(true)
        
        accuracies.append(np.mean(pred == true))
    
    return np.mean(accuracies)

def ROC_AUC_new(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError(f"Expect y_pred and y_true to have the same number of rows, but got {len(y_pred)} and {len(y_true)}.")

    aucs = []
    for pred, true in zip(y_pred, y_true):
        pred = np.array(pred)
        true = np.array(true)

        # Sort true labels by the descending order of predictions
        rank = np.argsort(-pred)
        true = true[rank]
        
        tps = np.cumsum(true)
        fps = np.cumsum(1 - true)
        
        tpr = tps / (tps[-1] + np.finfo(float).eps)
        fpr = fps / (fps[-1] + np.finfo(float).eps)
        
        auc = np.trapz(tpr, fpr)
        aucs.append(auc)
    
    return np.nanmean(aucs)