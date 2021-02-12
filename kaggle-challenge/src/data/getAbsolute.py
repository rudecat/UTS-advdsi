# -*- coding: utf-8 -*-
import logging
import pandas as pd
import joblib

def abs(df, name):
    """Convert the dataframe value to absolute:
    ***WARNING*** This module converts all negative values to positive values in dataframe.
    
    Parameters
    ----------
    df : dataframe
        Features of dataset
    name: string
        The name of the dataset, and it will be used as the filename of data dump

    Returns
    -------
    df : dataframe
        Converted dataframe with all values in absolute
    """

    logger = logging.getLogger(__name__)
    logger.info('turning dataframe '+name+ ' to be absolute value')

    df = df.abs()
    joblib.dump(df, "../data/processed/abs_"+name)
    return df
