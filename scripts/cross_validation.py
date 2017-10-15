import numpy as np
from scripts.ml_method import logistic_regression
from scripts.proj1_helpers import predict_labels

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k):
    
    mask = np.ones(x.shape[0], dtype = bool)
    mask[k_indices[k]] = False
    
    x_te = x[k_indices[k]]
    x_tr = x[mask]
    y_te = y[k_indices[k]]
    y_tr = y[mask]

    # ***************************************************
    
    losses, w = logistic_regression(y_tr, x_tr , np.zeros(x.shape[1]),300, 0.000005)
    
    acc_te = np.mean(predict_labels(w, x_te) == y_te)
    acc_tr = np.mean(predict_labels(w, x_tr) == y_tr)
    
    return acc_tr, acc_te, w
