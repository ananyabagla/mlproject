
import os
import pickle
from src.exception import CustomException
from src.logger import logging

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