import numpy as np
from scripts.helpers import sigmoid

def compute_gradient_mse(y, tx, w):
    """Function to compute gradient for given model w.

    Args:
        y  (numpy array): Matrix output of size N x 1.
        tx (numpy array): Matrix input of size N x D.
        w  (numpy array): Matrix weight (parameters of the model) of size D x 1.

    Returns:
        gradient (numpy array) : Matrix Gradient of size D x 1.
    """
    
    value_error = y - tx.dot(w)
    gradient    = -(np.transpose(tx).dot(value_error)) / (y.shape[0])
    return gradient

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - (y+1)/2)
    return grad

def compute_gradient_logistic_regularized(y, tx, w, lambda_):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - (y+1)/2)
    grad += 2 * y.shape[0] * lambda_ * w
    return grad