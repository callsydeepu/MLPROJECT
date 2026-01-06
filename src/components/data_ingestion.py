import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class DataIngestionConfig():
    artifacts_dir:str = PROJECT_ROOT / "Artifacts"
    dataset_path:str = r"C:\Users\WELCOME\Desktop\MLPROJECTS\notebook\data\stud.csv"
    train_file:str="train.csv"
    test_file:str="test.csv"
    raw_file:str="raw.csv"
    test_size:float = 0.2
    random_state:int =1
    target_col :str ="math_score"
    stratisfy :bool =False #disable for regression

class DataIngestion():
    def __init__(self,config:DataIngestionConfig=DataIngestionConfig()):
        self.config=config
    
    def _ensure_artifacts_dir(self):
        os.makedirs(self.config.artifacts_dir,exist_ok=True)
    def _timestamped_path(self,filename:str):
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.config.artifacts_dir,f"{ts}_{filename}")
    
    def _validate_dataset(self,df:pd.DataFrame):
        logging.info("Validating dataset structure...")

        if df.empty:
            raise ValueError("Data set is empty!!")
        
        if self.config.target_col not in df.columns:
            raise ValueError(
                f"target column '{self.config.target_col}' not found in dataset."
                f"Available columns are : {df.columns}"
            )
        
        missing=df.isna().sum()
        if missing.sum() >0:
            logging.warning(f"Dataset contains missing values:\n{missing}")

        
        #------main---------
    def initiate_dataingestion(self):
        logging.info("Data ingestion started")

        try:
            self._ensure_artifacts_dir()

            #--load__dataset

            if not os.path.exists(self.config.dataset_path):
                raise FileNotFoundError(f"dataset not found")
            
            df=pd.read_csv(self.config.dataset_path)
            logging.info(f"dataset loaded:Shape:{df.shape} ")

            self._validate_dataset(df)

            raw_path=self._timestamped_path(self.config.raw_file)
            df.to_csv(raw_path,index=False)

            logging.info(f"Raw dataset saved at {raw_path}")

            #__split__
            stratify_col=(
                df[self.config.target_col] if self.config.stratisfy else None
            )

            train_df,test_df=train_test_split(df,
                                              test_size=self.config.test_size,
                                              random_state=self.config.random_state,
                                              stratify=stratify_col)
            
            logging.info(
                f"Train/Test split completed "
                f"(train={train_df.shape}, test={test_df.shape})"
            )
            
            train_path=self._timestamped_path(self.config.train_file)
            test_path=self._timestamped_path(self.config.test_file)

            df.to_csv(train_path,index=False)
            df.to_csv(test_path,index=False)

            logging.info(f"Train data saved at {train_path}")
            logging.info(f"Test data saved at {test_path}")
            logging.info("Data ingestion completed successfully")

            return train_path,test_path
        except Exception as e:
            logging.error("DataIngestion failed")
            raise CustomException(e,sys)
        
    
if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_dataingestion()
        
