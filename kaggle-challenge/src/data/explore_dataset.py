# -*- coding: utf-8 -*-
import logging
import pandas as pd

def explore(filename):
    """Explore data:
    This module loads the csv file with pandas and print out the standard dataframe information.
    
    Parameters
    ----------
    filename : str
        The filename of the target file to be explored.

    Returns
    -------
    
    """

    logger = logging.getLogger(__name__)
    logger.info('explore data from path: data/raw/'+filename)

    pd.set_option('display.max_columns', None)
    df = pd.read_csv("../data/raw/"+filename)
    
    print("=== dataframe info ===")
    print(df.info(), end="\n")
    print("=== dataframe shape ===")
    print(df.shape, end="\n")
    if 'TARGET_5Yrs' in df:
        print("=== Target Value Count ===")
        print(df['TARGET_5Yrs'].value_counts(), end="\n")
    print("=== dataframe describe ===")
    print(df.describe(include='all'), end="\n")
