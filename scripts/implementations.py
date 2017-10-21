import numpy as np
from scripts.helpers import batch_iter, sigmoid
from scripts.costs import compute_mse, compute_rmse, compute_loss_logistic, compute_loss_logistic_regularized
from scripts.gradients import compute_gradient_mse, compute_gradient_logistic, compute_gradient_logistic_regularized

# 1. Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Function to perfom gradient descent algorithm.

    Args:
        y         (numpy array): Matrix output of size N x 1.
        tx        (numpy array): Matrix input of size N x D.
        initial_w (numpy array): Initial matrix weight (parameters of the model) of size D x 1.
        max_iters (int)        : Maximum iteration for gradient descent training.
        gamma     (int)        : Learning rate constant.

    Returns:
        losses (list)       : List of lost value per iteration within the training.
        ws     (list)       : List of updated weight matrix per iteration within the training.
        w      (numpy array): Final weight matrix by gradient descent.
    """
    
    w = initial_w
    
    for n_iter in range(max_iters):

        loss = compute_mse(y, tx, w)
        grad = compute_gradient_mse(y,tx,w)

        w = w - gamma * grad
        
    return w, loss

# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for minibatch_y, minibatch_x in batch_iter(y,tx,32):
        loss = compute_mse(minibatch_y, minibatch_x, w)
        grad = compute_gradient_mse(minibatch_y, minibatch_x, w)
        
        w = w - gamma * grad
        
    return w, loss


# Least squares regression using normal equations
def least_squares(y, tx):
    xt = tx.transpose()
    w = np.linalg.solve(xt.dot(tx), xt.dot(y))
    loss = compute_rmse(y, tx, w)
    
    return loss, w

# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    xt = tx.transpose()
    weights = np.linalg.solve(xt.dot(tx) + lambda_ * (2 * y.shape[0]) * np.identity(xt.shape[0]), xt.dot(y))

    return weights

# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    loss = compute_loss_logistic(y, tx, w)
    grad = compute_gradient_logistic(y,tx,w)

    w = w - gamma * grad

    for n_iter in range(max_iters):
        # need change
        loss = compute_loss_logistic(y, tx, w)
        grad = compute_gradient_logistic(y,tx,w)

        w = w - gamma * grad

    return w, loss

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    w = initial_w
    
    for n_iter in range(max_iters):
        # need change
        loss = compute_loss_logistic_regularized(y, tx, w, lambda_)
        grad = compute_gradient_logistic_regularized(y,tx,w, lambda_)

        w = w - gamma * grad

    return w, loss