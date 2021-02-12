# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def smote(X,y,strategy=0.2):
    """SMOTE:
    To tackle imbalanced data, this function helps creating synthetic samples from minority.
    
    Parameters
    ----------
    X : dataframe or array
        Features of dataset
    y : dataframe or array
        Target value of dataset
    strategy : float
        The desired ratio of minority sample

    Returns
    -------
    X : dataframe or array
        Features of new dataset
    y : dataframe or array
        Target value of new dataset
    """
    # for reproducibility
    seed = 100
    # SMOTE number of neighbors
    k = 5
    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    X,y = sm.fit_resample(X,y)
    return X,y

def random_under(X,y,strategy=0.5):
    """Random Undersampling:
    To tackle imbalanced data, this function helps removing samples from majority.
    
    Parameters
    ----------
    X : dataframe or array
        Features of dataset
    y : dataframe or array
        Target value of dataset
    strategy : float
        The desired ratio of majority sample

    Returns
    -------
    X : dataframe or array
        Features of new dataset
    y : dataframe or array
        Target value of new dataset
    """

    under = RandomUnderSampler(sampling_strategy=strategy)
    X,y = under.fit_resample(X,y)
    return X,y


def oversample(df):
    """Oversampling:
    To tackle imbalanced data, this function helps upsampling the minority.
    
    Parameters
    ----------
    df : dataframe
        All dataset (including features and target value)

    Returns
    -------
    df : dataframe
        All dataset with additional duplicate minority samples.
    """
    
    logger = logging.getLogger(__name__)
    logger.info('Oversample the dataset to handle imbalanced data')

    df_1 = df.loc[df['TARGET_5Yrs']==1]
    df_1_len = len(df_1.index)
    df_0 = df.loc[df['TARGET_5Yrs']==0]
    df_0_len = len(df_0.index)

    if ( df_1_len > df_0_len ):
        df_0 = resample(df_0, replace=True, n_samples=df_1_len, random_state=123)
    else:
        df_1 = resample(df_1, replace=True, n_samples=df_0_len, random_state=123)

    df = df_1.append(df_0, ignore_index=True)
    return df
