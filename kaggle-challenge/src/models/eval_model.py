from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, plot_confusion_matrix
from sklearn.base import is_regressor
import numpy as np
import matplotlib.pyplot as plt

def eval_model(classifier, X_train, y_train, X_val, y_val, show=True):
  """Evaluate Model:
    This module train the model based on passed in classifier and training set, 
    and then use valuation set to evaluate the performance of trained model.
    It can select to print out or not all related metrics for investigation.

    Parameters
    ----------
    classifier : sklearn classifier
        classifier to be evaluated
    X_train : dataframe or array
        Features of training set
    y_train : dataframe or array
        Target value of training set
    X_val : dataframe or array
        Features of valuation set
    y_val : dataframe or array
        Target value of valuation set
    show : boolean
        If metric should be shown or not

    Returns
    -------
    model
        Trained model based on classifier and training set data
    roc_score_training
        ROC AUC score of training set
    roc_score_val
        ROC AUC score of valuation set
    """

  model = classifier.fit(X_train,y_train)
  roc_score_training = get_performance(model, X_train, y_train, "Train", show)
  roc_score_val = get_performance(model, X_val, y_val, "Validate", show)  

  return model, roc_score_training, roc_score_val

def get_performance(mod, xvar, yvar, runtype, show):
  """Evaluate the performance of the model:
    This function uses valuation set to evaluate the performance of trained model.
    It can select to print out or not all related metrics for investigation.

    Parameters
    ----------
    mod : sklearn classifier
        Trained model
    xvar : dataframe or array
        Features
    yvar : dataframe or array
        Target value
    runtype : string
        Enum : [Train, Validate] 
        This value is used to label printed metrics for better understanding
    show : boolean
        If metric should be shown or not

    TODO: Add more relavent scores 
    TODO: Have a switch to select / remove metrics
    TODO: More Plots/Graphs for clear indications

    Returns
    -------
    mod_roc_score
        ROC AUC score of dataset
        
    """
  if is_regressor(mod):
    convert_ratio = np.vectorize(lambda x: 1 if x > 0.5 else 0 )
    mod_pred_proba = mod.predict(xvar)
    mod_pred = convert_ratio(mod_pred_proba)
  else:
    mod_pred = mod.predict(xvar)
    mod_pred_proba = mod.predict_proba(xvar)[:, 1]

  mod_roc_score = roc_auc_score(yvar, mod_pred_proba)
  
  if show:
    print('Accuracy Score: ',accuracy_score(yvar,mod_pred),' F1 Score ',f1_score(yvar,mod_pred),' Recall Score ', recall_score(yvar,mod_pred), ' R2 Score ',mod.score(xvar, yvar),' ROC_AUC_SCORE ', mod_roc_score,'(',runtype,')')
    disp = plot_confusion_matrix(mod, xvar, yvar, cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion matrix '+runtype)
  
  return mod_roc_score
