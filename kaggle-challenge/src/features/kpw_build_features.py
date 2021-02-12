# -*- coding: utf-8 -*-
import logging

def build(X):
    """Build Features:
    This module is to create new features from existing features

    Parameters
    ----------
    X : dataframe
        Features of dataset
    
    Returns
    -------
    X
        New dataframe with new features
    """
    logger = logging.getLogger(__name__)
    logger.info('Build features')

    X = X.drop(columns=['FG%','3P%','FT%'])

    X['FG%'] = X.apply(lambda row: row['FGM'] / row['FGA'] if row['FGA'] > 0 else 0, axis = 1)
    X['FGM'] = X.apply(lambda row: row['FGM'] if row['FGM'] <= row['FGA'] else row['FGA'], axis = 1)
    X['3P%'] = X.apply(lambda row: row['3P Made'] / row['3PA'] if row['3PA'] > 0 else 0, axis = 1)
    X['3P Made'] = X.apply(lambda row: row['3P Made'] if row['3P Made'] <= row['3PA'] else row['3PA'], axis = 1)
    X['FT%'] = X.apply(lambda row: row['FTM'] / row['FTA'] if row['FTA'] > 0 else 0, axis = 1)
    X['FTM'] = X.apply(lambda row: row['FTM'] if row['FTM'] <= row['FTA'] else row['FTA'], axis = 1)
    # *_small features are created when the stat is in lower 25%
    X['GP_small'] = X.apply(lambda row: 1 if row['GP'] < 51 else 0, axis = 1)
    X['MIN_small'] = X.apply(lambda row: 1 if row['MIN'] <= 12 else 0, axis = 1)
    X['PTS_small'] = X.apply(lambda row: 1 if row['PTS'] <= 4.2 else 0, axis = 1)
    X['FTM_small'] = X.apply(lambda row: 1 if row['FTM'] <= 0.7 else 0, axis = 1)
    X['OREB_small'] = X.apply(lambda row: 1 if row['OREB'] <= 0.5 else 0, axis = 1)
    X['AST_small'] = X.apply(lambda row: 1 if row['AST'] <= 0.6 else 0, axis = 1)
    X['BLK_small'] = X.apply(lambda row: 1 if row['BLK'] <= 0.1 else 0, axis = 1)
    # BadStats combines all *_small features and then average them to form an indication.
    # BadStats = 1 means a player performed poorly across the board
    # BadStats = 0 means a player performed not-poorly across the board
    X['BadStats'] = X.apply(lambda row: (row['GP_small'] + row['MIN_small'] + row['PTS_small'] + row['FTM_small'] + row['OREB_small'] + row['AST_small'] + row['BLK_small']) / 7, axis = 1)
    X['NotActive'] = X.apply(lambda row: 1 if (row['FGA']*row['3PA']*row['FTA'] == 0) else 0, axis = 1)
    X['NotTeamPlayer'] = X.apply(lambda row: 1 if (row['AST']*row['BLK']*row['DREB'] == 0) else 0, axis = 1)
    # X = X.drop(columns=['FGM','FGA','3P Made','3PA','FTA','DREB','REB','TOV','STL'])
    X = X.drop(columns=['GP_small','MIN_small','PTS_small','FTM_small','OREB_small','AST_small','BLK_small'])

    return X
