from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
import sys
from src.utils import save_object
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
if __name__ == "__main__":
    try:
        logging.info("Training pipeline started")
        
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        logging.info("Training pipeline completed successfully")
        logging.info(f"Preprocessor saved at: {preprocessor_path}")

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training pipeline completed successfully")

        
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise