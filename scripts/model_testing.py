import numpy as np
from scripts.proj1_helpers import predict_labels
from scripts.split_data import split_data
from scripts.ml_method import reg_logistic_regression
from scripts.ml_method import logistic_regression

def curve_lambda(y, stand_x, ratio, init_, step_, lambda_):
    print('lambda, train_error, test_error')
    for lambda_ in np.arange(init_, final_, step_):
        x_tr, x_te, y_tr, y_te = split_data(stand_x, y, ratio, myseed=1)
        losses, w = reg_logistic_regression(y_tr, x_tr , np.zeros(stand_x.shape[1]),300, 0.000005, lambda_)
        y_pred = predict_labels(w, x_tr)
        y_pred1 =  predict_labels(w, x_te)
        print(lambda_, ',', np.mean(y_pred == y_tr),',', np.mean(y_pred1 == y_te))

def curve_size_train_set(y, stand_x, init_, step_, final_):
    print('train_error, test_error')
    for ratio in np.arange(init_, final_, step_):
        x_tr, x_te, y_tr, y_te = split_data(stand_x, y, ratio, myseed=1)
        losses, w = logistic_regression(y_tr, x_tr , np.zeros(stand_x.shape[1]),300, 0.000005)
        y_pred = predict_labels(w, x_tr)
        y_pred1 =  predict_labels(w, x_te)
        print(ratio, ',', np.mean(y_pred == y_tr),',', np.mean(y_pred1 == y_te))