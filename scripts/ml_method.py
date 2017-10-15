import numpy as np
from scripts.costs import *

# Compute gradient for MSE
def compute_gradient(y, tx, w):

	grad = np.zeros(2)   
	v_error = y - tx.dot(w)

	grad[0] = -np.sum(v_error)/y.shape[0]
	grad[1] = -1/y.shape[0] * np.sum(v_error * tx[:,1])
	
	return grad

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):

	ws = [initial_w]
	losses = []
	w = initial_w
	
	for n_iter in range(max_iters):

		loss = compute_mse(y, tx, w)
		grad = compute_gradient(y,tx,w)

		w = w - gamma * grad
		ws.append(w)
		losses.append(loss)
		print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

	return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
	"""
	Generate a minibatch iterator for a dataset.
	Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
	Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
	Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
	Example of use :
	for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
		<DO-SOMETHING>
	"""
	data_size = len(y)

	if shuffle:
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffled_y = y[shuffle_indices]
		shuffled_tx = tx[shuffle_indices]
	else:
		shuffled_y = y
		shuffled_tx = tx
	for batch_num in range(num_batches):
		start_index = batch_num * batch_size
		end_index = min((batch_num + 1) * batch_size, data_size)
		if start_index != end_index:
			yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	ws = [initial_w]
	losses = []
	w = initial_w
	
	n_iter = 0
	batch = batch_iter(y, tx, batch_size, num_batches = max_iters)
	
	for b in batch:
		loss = compute_mse(y, tx, w)          
		grad = compute_gradient(b[0], b[1],w)
		w = w - gamma * grad

		ws.append(w)
		losses.append(loss)
		print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
		n_iter += 1
		
	return losses, ws

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
	loss = compute_rmse(y, tx, w)

	return loss, weights

# def sigmoid(tx, w):
# 	z = np.exp(-tx.dot(w))
# 	return 1.0/(1.0 + z)

# def compute_gradient_logistic(y, tx, w):
#     return tx.T.dot(sigmoid(tx, w) - y)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = ((y + 1)/2).T.dot(np.log(pred)) + ((1 - y)/2).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - (y+1)/2)
    return grad

# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
	ws = [initial_w]
	losses = []
	w = initial_w
	
	for n_iter in range(max_iters):

		# need change
		loss = compute_loss_logistic(y, tx, w)
		grad = compute_gradient_logistic(y,tx,w)

		w = w - gamma * grad
		ws.append(w)
		losses.append(loss)
		# print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			  # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

	return losses, np.array(w)

def compute_loss_logistic_regularized(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = ((y + 1)/2).T.dot(np.log(pred)) + ((1 - y)/2).T.dot(np.log(1 - pred))
    loss -= y.shape[0] * lambda_ * np.inner(w, w)
    return np.squeeze(- loss)

def compute_gradient_logistic_regularized(y, tx, w, lambda_):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - (y+1)/2)
    grad += 2 * y.shape[0] * lambda_ * w
    return grad

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
	ws = [initial_w]
	losses = []
	w = initial_w
	
	for n_iter in range(max_iters):

		# need change
		loss = compute_loss_logistic_regularized(y, tx, w, lambda_)
		grad = compute_gradient_logistic_regularized(y,tx,w, lambda_)

		w = w - gamma * grad
		ws.append(w)
		losses.append(loss)
		# print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
			  # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

	return losses, np.array(w)
