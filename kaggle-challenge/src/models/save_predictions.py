import pandas as pd
import numpy as np

def save_predictions(name:str, classifier: any, data:np.ndarray) -> np.ndarray:
    '''Make predictions and save to disk
    
    Usage:
        predictions = save_predictions('au-ron_week-1_0.70974.csv', log_reg, test_data)
    
    Parameters:
        name (str): Filename for output
        classifier (model): Trained model
        data (np.ndarray): Test data, with transformations already applied
        
    Returns:
        predictions (np.ndarray): Array of Id and prediction probabilities
    '''

    file_name = f'../data/predictions/{name}'
    predictions = classifier.predict_proba(data)

    predictions[:, 0] = np.arange(3799).astype(int)

    np.savetxt(file_name, predictions, delimiter=',', header='Id,TARGET_5Yrs', comments='')
    print(f'Predictions saved to {file_name}')
    csv = pd.read_csv(file_name)
    csv['Id'] = csv['Id'].astype(int)

    return predictions
