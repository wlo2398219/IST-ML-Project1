<<<<<<< e188284032459f1acf48519b3d0d8ea2223f45f7
import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)
=======
# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

# *******************
# General Setting
# y: N x 1 vector
# tx: N x D matrix
# w: D x 1 vector
# *******************

def compute_loss(y, tx, w):
 
    v_error = y - tx.dot(w)
    error = np.inner(v_error, v_error)/(2 * y.shape[0]) 
    return error
 

def compute_mse(y, tx, w):

    v_error = y - tx.dot(w)
    v_error = y - y_expect
    error = np.inner(v_error, v_error)/(2 * y.shape[0])
    return error

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))
>>>>>>> Add costs.py
