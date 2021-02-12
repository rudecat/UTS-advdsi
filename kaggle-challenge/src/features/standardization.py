# -*- coding: utf-8 -*-
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def standardScale(X_train, X_val):
    """StandardScaling Standardization:
    This module is to fit a StandardScaler with training set, and transform both training and valuation sets.
    Then return the trained scaler, transformed training set, and transformed valuation/test set.

    Parameters
    ----------
    X_train : dataframe or array
        Features of training set
    X_val : dataframe or array
        Features of valuation/test set

    Returns
    -------
    scaler
        Trained scaler based on training set
    X_train
        transformed training set
    X_val
        transformed valuation/test set
    """
    logger = logging.getLogger(__name__)
    logger.info('Standard scaling on training, valuation and test sets')

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return scaler, X_train, X_val


def minMaxScale(X_train, X_val):
    """MinMaxScaler Standardization:
    This module is to fit a MinMaxScaler with training set, and transform both training and valuation sets.
    Then return the trained scaler, transformed training set, and transformed valuation/test set.

    Parameters
    ----------
    X_train : dataframe or array
        Features of training set
    X_val : dataframe or array
        Features of valuation/test set

    Returns
    -------
    scaler
        Trained scaler based on training set
    X_train
        transformed training set
    X_val
        transformed valuation/test set
    """
    logger = logging.getLogger(__name__)
    logger.info('Min-max scaling on training, valuation and test sets')

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return scaler, X_train, X_val
