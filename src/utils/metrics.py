import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def logistic_regression(y, y_pred):
    # Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)
    return y_pred_logistic


def calculate_rmse(y_pred, y, fit_scale=None, eps=1e-8):
    if fit_scale is not None:
        y_pred = logistic_regression(y, y_pred)
    return np.sqrt(np.mean((y_pred - y) ** 2) + eps)


def calculate_plcc(y_pred, y, fit_scale=None):
    if fit_scale is not None:
        y_pred = logistic_regression(y, y_pred)
    return stats.pearsonr(y_pred, y)[0]


def calculate_srcc(y_pred, y):
    return stats.spearmanr(y_pred, y)[0]


def calculate_krcc(y_pred, y):
    return stats.kendalltau(y_pred, y)[0]
