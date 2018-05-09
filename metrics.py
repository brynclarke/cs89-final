from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, log_loss

def mse(y_true, y_pred):
    return mean_squared_error(y_pred, y_true)

def rmse(y_true, y_pred):
    return sqrt(mse(y_pred, y_true))

def mae(y_true, y_pred):
    return mean_absolute_error(y_pred, y_true)

def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def xent(y_true, y_pred):
    return log_loss(y_true, y_pred)

def gini(y_true, y_pred):
    return 2*roc_auc_score(y_true, y_pred) - 1