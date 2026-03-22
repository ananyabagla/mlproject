
import os
import pickle
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True) 
        logging.info(f"Directory created successfully at {dir_path} for saving object")
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            logging.info(f"Object saved successfully at {file_path}")
            
    except Exception as e:
        logging.error(f"Error saving object at {file_path}: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            logging.info(f"Object loaded successfully from {file_path}")
            return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,param_grids):
    try:
        report = {}
        logging.info("Evaluating models started")
        for i in range(len(models)):
            model = list(models.values())[i]
            param_grids_for_model = param_grids.get(list(models.keys())[i], {})
            if param_grids_for_model:
                logging.info(f"Hyperparameter tuning for {list(models.keys())[i]} with parameters: {param_grids_for_model}")
                grid_search = GridSearchCV(estimator=model, param_grid=param_grids_for_model, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                logging.info(f"Best hyperparameters for {list(models.keys())[i]}: {best_params}")
                model.set_params(**best_params)  # Update model with best hyperparameters
                model.fit(X_train, y_train)  # Fit the model with the best hyperparameters
                y_train_pred = model.predict(X_train)  # Predict on training data
                y_test_pred = model.predict(X_test)  # Predict on test data
                train_model_score = r2_score(y_train, y_train_pred)  # Calculate R2
                test_model_score = r2_score(y_test, y_test_pred)  # Calculate R2
                report[list(models.keys())[i]] = test_model_score


           
            
                logging.info(f"Model {list(models.keys())[i]} evaluated with R2 score: {test_model_score}")
        return report
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            logging.info(f"Object loaded successfully from {file_path}")
            return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)