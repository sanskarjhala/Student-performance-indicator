import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomExceptiion
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)
        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)

    except Exception as e:
        raise CustomExceptiion(e,sys)
    
def evalute_models(X_train , y_train , X_test , y_test , models , params):
    try:
        report = {}

        for i in range(len(models)):  # Directly iterate over models dictionary
            model_name = list(models.keys())[i]  # Get model name
            model = models[model_name]  # Get model object
            param = params[model_name]

            rs = RandomizedSearchCV(model ,param_distributions=param, cv=3 , scoring='r2',random_state=42 , n_jobs=-1)
            rs.fit(X_train , y_train)

            logging.info(f"Best hyperParameters found {rs.cv_results_}")
            
            #updating the model with hyperparameter tunning
            model.set_params(**rs.best_params_)
            model.fit(X_train , y_train)

            y_train_pred = model.predict(X_train)  
            y_test_pred = model.predict(X_test)  

            train_model_score = r2_score(y_train, y_train_pred)  
            test_model_score = r2_score(y_test, y_test_pred)  

            report[model_name] = test_model_score  # Store in dictionary

        return report
    
    except Exception as e:
        raise CustomExceptiion()

def load_object(file_path):
    try:
        with open(file_path , 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomExceptiion(e, sys)