import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from pathlib import Path

@dataclass
class ModelTrainingConfig:
    model_file_name:str="model.pkl"

class ModelTrainer:
    def __init__(self,config:ModelTrainingConfig=ModelTrainingConfig()):
        self.config=config
    def initiate_model_trainer(
            self,
            train_arr:np.ndarray,
            test_arr:np.ndarray,
            run_dir:Path
    ):
        try:
            logging.info("Starting model training!!")

            #split features and target
            X_train=train_arr[:, :-1]
            y_train=train_arr[:,-1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")

            #train baseline model
            model=LinearRegression()
            model.fit(X_train,y_train)

            #evaluate

            y_pred=model.predict(X_test)
            score=r2_score(y_test,y_pred)

            logging.info(f"R2 score:{score}")

            #save model
            model_path=run_dir/self.config.model_file_name
            save_object(model_path,model)

            logging.info(f"model saved at {model_path}")

            return score
        except Exception as e:
            logging.error("Model training failed")
            raise CustomException(e, sys)
        