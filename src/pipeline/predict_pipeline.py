import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import sys

class PredictPipeline:
    def __init__(self):
        logging.info("PredictPipeline initialized")

    def predict(self, features):
        try:
            logging.info("--- STARTING PREDICTION PROCESS ---")
            
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            
            logging.info(f"Attempting to load preprocessor from: {preprocessor_path}")
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Preprocessor loaded successfully")
            
            logging.info(f"Attempting to load model from: {model_path}")
            model = load_object(file_path=model_path)
            logging.info("Model loaded successfully")
            
            logging.info("Input features received. Head of dataframe:")
            logging.info(f"\n{features.head()}")

            logging.info("Starting feature transformation (scaling/encoding)...")
            features_scaled = preprocessor.transform(features)
            logging.info("Transformation complete. Scaled data shape: " + str(features_scaled.shape))
            
            logging.info("Feeding scaled data into the model for prediction...")
            prediction = model.predict(features_scaled)
            logging.info(f"Raw prediction result: {prediction}")
            
            return prediction[0]
        
        except Exception as e:
            logging.error(f"CRITICAL ERROR IN PIPELINE: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        # Dynamically set attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        logging.info("CustomData object created with form inputs")

    def get_data_as_dataframe(self):
        try:
            data_dict = {k: [v] for k, v in self.__dict__.items()}
            df = pd.DataFrame(data_dict)
            logging.info("Data successfully converted to DataFrame for prediction")
            return df
        except Exception as e:
            logging.error(f"Error converting data to DataFrame: {e}")
            raise CustomException(e, sys)