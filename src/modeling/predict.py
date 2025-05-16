import pickle
import numpy as np


def predict(X_input: np.array, task_type: str, pred_threshold: float = None) -> np.array:
    """
    Predict outcomes using the trained ANN model.

    Args:
        X_input (np.array): Input features for prediction.
        task_type (str): Type of task - "Binary Classification" or "Regression".
        pred_threshold (float, optional): Threshold for binary classification. Ignored for regression.

    Returns:
        np.array: Predicted outcomes.
    """    
    
    model_file = "models/ann_model.pkl"
    with open(model_file, "rb") as file:
        ann_model = pickle.load(file=file)

    y_pred = ann_model.predict(X_input)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    y_pred = ann_model.predict(X_input)
    
    if task_type == 'Binary Classification':
        if pred_threshold is None:
            raise ValueError('pred_threshold must be specified for binary classification')
        y_pred = (y_pred > pred_threshold).astype(int)
        
    elif task_type == 'Regression':
        pass
    else:
        raise ValueError('Invalid task_type. Please choose "Binary Classification" or "Regression"')

    return y_pred
