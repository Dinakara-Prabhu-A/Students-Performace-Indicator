import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill 
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return a performance report.
    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        models: A dictionary of model names and their instances.
    """
    report = {}
    for model_name, model_instance in models.items():
        try:
            logging.info(f"Evaluating model: {model_name}")
            model_instance.fit(X_train, y_train)
            test_model_score = model_instance.score(X_test, y_test)  # Use only test score
            report[model_name] = test_model_score
        except Exception as e:
            raise CustomException(f"Error evaluating model {model_name}: {str(e)}", sys)
    return report


