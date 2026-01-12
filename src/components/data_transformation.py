import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass,field
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    target_col:str="math_score"
    drop_columns: list = field(default_factory=lambda: ["reading_score", "writing_score"])
    #Create a NEW list every time a new config object is created.”
    preprocessor_file_name:str="preprocessor.pkl"


class DataTransformation:
    def __init__(self,config:DataTransformationConfig=DataTransformationConfig()):
        self.config=config

    def initiate_data_transformation(
            self,
            train_path:Path,
            test_path:Path,
            run_dir:Path
    ):

        try:
            logging.info("Starting data transformation")

            #load data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            #sep feature and target
            X_train=train_df.drop(columns=[self.config.target_col])
            y_train=train_df[self.config.target_col]


            X_test=test_df.drop(columns=[self.config.target_col])
            y_test=test_df[self.config.target_col]

            #drop leakage
            if self.config.drop_columns:
                X_train = X_train.drop(columns=self.config.drop_columns, errors="ignore")
                X_test = X_test.drop(columns=self.config.drop_columns, errors="ignore")

                logging.info(f"Dropped columns: {self.config.drop_columns}")

            #feature type detection

            numeric_features=X_train.select_dtypes(include=["int64","float64"]).columns
            cat_features=X_train.select_dtypes(include=["object"]).columns

            logging.info(f"Numeric features: {list(numeric_features)}")
            logging.info(f"Categorical features: {list(cat_features)}")

            #pipelines

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ohe",OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            preprocessor=ColumnTransformer(
                transformers=[
                    ("num",num_pipeline,numeric_features),
                    ("cat",cat_pipeline,cat_features)
                ]
            )

            #fit and train

            X_train_processed=preprocessor.fit_transform(X_train)
            X_test_processed=preprocessor.transform(X_test)

            #combine X and y-->beacause model_trainer needs only one obj 
            #X_train_processed=(800,12),y_train(800,1)
            #combined=(800,13)

            # handle sparse matrix output
            if hasattr(X_train_processed, "toarray"):
                X_train_processed = X_train_processed.toarray()
                X_test_processed = X_test_processed.toarray()


            #c_ only works on numpy array but ohe gives sparse mat
            train_arr = np.c_[X_train_processed, y_train.to_numpy()]
            test_arr  = np.c_[X_test_processed, y_test.to_numpy()]


            #save processor

            preprocessor_path=run_dir/self.config.preprocessor_file_name
            save_object(preprocessor_path,preprocessor)

            logging.info(f"Preprocessor saved at {preprocessor_path}")
            logging.info("Data transformation completed")

            return train_arr,test_arr,preprocessor_path
        
        except Exception as e:
            logging.error("Data transformation failed")
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestion
#     from src.components.data_transformation import (
#         DataTransformation,
#         DataTransformationConfig
#     )
#     from src.logger import logging

#     logging.info("DEBUG MODE: Running data transformation standalone")

#     # Step 1: Run ingestion to get required inputs
#     ingestion = DataIngestion()
#     train_path, test_path, run_dir = ingestion.initiate_data_ingestion()

#     # Step 2: Run transformation
#     transformer = DataTransformation(
#         DataTransformationConfig(
#             target_col="math_score",
#             drop_columns=["reading_score", "writing_score"]
#         )
#     )

#     train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
#         train_path=train_path,
#         test_path=test_path,
#         run_dir=run_dir
#     )

#     logging.info("DEBUG MODE: Data transformation completed successfully")

#     print("Train array shape:", train_arr.shape)
#     print("Test array shape:", test_arr.shape)
#     print("Preprocessor saved at:", preprocessor_path)
