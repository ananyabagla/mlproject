import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.utils import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                'Linear Regression': LinearRegression(),
                'Lasso':Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'Support Vector Regressor': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'XGBoost Regressor': XGBRegressor(random_state=42),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42)
            }
            logging.info("Models defined successfully for training")
            logging.info("model training started")

            model_report :dict = evaluate_models(X_train = X_train, y_train=y_train, X_test=X_test, y_test = y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model is {best_model_name} with R2 score: {best_model_score}")
            if best_model_score<0.6:
                logging.info("No best model found with R2 score greater than 0.6")
                raise CustomException("No best model found with R2 score greater than 0.6", sys)

            logging.info("Model training completed successfully")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")
            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            return r2
        except Exception as e:
            logging.info("Error occurred in Model Training")
            raise CustomException(e, sys)