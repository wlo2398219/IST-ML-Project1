import numpy as np

def compute_mse(y, tx, w):
    """Function to compute mean square error (MSE).

    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.

    Returns:
        mse (float, scalar) : The mean square error (MSE) for given model w.
    """
    
    value_error = y - tx.dot(w)
    mse         = np.inner(value_error, value_error)/(2 * y.shape[0]) 
    return mse

def compute_mae(y, tx, w):
    """Function to compute mean absolute error (MAE).

    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.

    Returns:
        mae (float, scalar) : The mean absolute error (MAE) for given model w.
    """
    
    value_error = y - tx.dot(w)
    mae         = np.sum(np.fabs(value_error))/(y.shape[0])
    return mae

def compute_rmse(y, tx, w):
    """Function to compute root mean square error (RMSE).

    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.

    Returns:
        rmse (float, scalar) : The root mean square error (RMSE) for given model w.
    """
    
    rmse = np.sqrt(2*compute_mse(y, tx, w))
    return rmse
