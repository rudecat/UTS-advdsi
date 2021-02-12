import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.features import standardization
from src.models import eval_model

def cv(classifier, X, y):
    """Cross Validatioin Module:
        This module splits complete dataset into 5 stratified folds, and do below steps
        1. Preprocessing - Standardization based on sub-training set
        2. Loop through 5 folds to perform Model Evaluation, and record all 5 times ROC AUC Score
        3. Average all 5 times ROC AUC Score, and return for comparison

    Parameters
    ----------
    classifier : sklearn classifier
        classifier to be evaluated
    X : dataframe or array
        Features of complete training set
    y : dataframe or array
        Target value of complete training set

    Returns
    -------
    roc_training_avg
        Average ROC AUC score of sub-training set
    roc_val_avg
        Average ROC AUC score of sub-val set
    """

    n = 5
    skf = StratifiedKFold(n_splits = n)
    roc_training = np.array([])
    roc_val = np.array([])

    for train_index, val_index in skf.split(X, y):
        X_train = X[train_index]
        X_val = X[val_index]
        sc, X_train, X_val = standardization.standardScale(X_train, X_val)
        y_train, y_val = y[train_index], y[val_index]

        model, roc_score_training, roc_score_val = eval_model.eval_model(classifier, X_train, y_train, X_val, y_val, show=False)
        roc_training = np.append(roc_training,roc_score_training)
        roc_val = np.append(roc_val, roc_score_val)
    
    roc_training_avg = np.average(roc_training)
    roc_val_avg = np.average(roc_val)

    print("Avg ROC AUC score of training set is: "+str(roc_training_avg))
    print("Avg ROC AUC score of valuation set is: "+str(roc_val_avg))

    return roc_training_avg, roc_val_avg
