import numpy as np

# *******************
# General Setting
# y: N x 1 vector
# tx: N x D matrix
# w: D x 1 vector
# *******************

def compute_mse(y, tx, w):

    v_error = y - tx.dot(w)
    error = np.inner(v_error, v_error)/(2 * y.shape[0]) 
    return error

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

# def compute_loss_logistic(y, tx, w):
#     return np.log(1 + np.exp(tx.dot(w))).sum() - y.T.dot(tx.dot(w))