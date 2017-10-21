import numpy as np
from scripts.logistic_regression import logistic_regression, reg_logistic_regression

def build_k_indices(y, k_fold, seed):
    """Function to build k indices for k-fold.

    Args:
        y      (numpy array): Matrix output of size N x 1.
        k_fold (int)        : The value k, of k-fold cross validation.
        seed   (int)        : Integer value to seed the random generator.

    Returns:
        k_indices (numpy array) : Matrix of K x floor(N/K), as indices for unbiased test error of K-cross Validation.
    """

    num_row    = y.shape[0]
    interval   = int(num_row / k_fold)
    np.random.seed(seed)
    indices    = np.random.permutation(num_row)
    k_indices  = np.array[indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return k_indices

def cross_validation(y, x, k_indices, k, max_iter, lambda_):
    """Function to compute loss of training and error based on cross validation.

    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        k_indices (numpy array): Matrix of K x floor(N/K), as indices for unbiased test error of K-cross Validation.
        k         (int)        : Integer value to seed the random generator.
        max_iter  (int)        : The value k, of k-fold cross validation.
        lambda_   (int)        : Integer value to seed the random generator.

    Returns:
        k_indices (numpy array) : Matrix of K x floor(N/K), as indices for unbiased test error of K-cross Validation.
    """
    
    train_set_indices               = np.ones(x.shape[0], dtype = bool)
    train_set_indices[k_indices[k]] = False
    
    x_test  = x[k_indices[k]]
    y_test  = y[k_indices[k]]
    
    x_train = x[train_set_indices]
    y_train = y[train_set_indices]

    losses, w = reg_logistic_regression(y_train, x_train, np.zeros(x.shape[1]), max_iter, 0.000002, lambda_)
    acc_te    = np.mean(predict_labels(w, x_test) == y_test)
    acc_tr    = np.mean(predict_labels(w, x_train) == y_train)
    
    return acc_tr, acc_te, w, losses[max_iter - 1]
