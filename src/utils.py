
import os
import pickle
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.metrics import r2_score
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
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        logging.info("Evaluating models started")
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train the model
            y_test_pred = model.predict(X_test)  # Predict on test data
            test_model_score = r2_score(y_test, y_test_pred)  # Calculate R2 score
            report[list(models.keys())[i]] = test_model_score  # Store the score in the report
            logging.info(f"Model {list(models.keys())[i]} evaluated with R2 score: {test_model_score}")
        return report
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)