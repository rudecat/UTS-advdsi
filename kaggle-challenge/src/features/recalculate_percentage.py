# -*- coding: utf-8 -*-
import logging
import pandas as pd

def replaceAll(df):
    """Recalculate Percentage:
    Replace all percentage with newly calculated value FG%, 3P% and FT%

    Parameters
    ----------
    df : dataframe
        Features of dataset

    Returns
    -------
    df : dataframe
        Features of dataset with recalculated percentage features
    """
    
    logger = logging.getLogger(__name__)
    logger.info('replace all percentage with newly calculated value FG%, 3P% and FT%')

    df = df.drop(columns=['FG%','3P%','FT%'])

    df['FG%'] = df.apply(lambda row: row['FGM'] / row['FGA'] if row['FGA'] > 0 else 0, axis = 1)
    df['3P%'] = df.apply(lambda row: row['3P Made'] / row['3PA'] if row['3PA'] > 0 else 0, axis = 1)
    df['FT%'] = df.apply(lambda row: row['FTM'] / row['FTA'] if row['FTA'] > 0 else 0, axis = 1)

    return df
