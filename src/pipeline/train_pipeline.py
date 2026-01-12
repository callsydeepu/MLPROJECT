import sys
import numpy as np
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation  import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

from dataclasses import field

def run_pipeline():
    logging.info("Training pipeline started")

    #data ingestion
    ingestion=DataIngestion()
    train_path,test_path,run_dir=ingestion.initiate_data_ingestion()

    #data transformation

    transformer=DataTransformation(
        DataTransformationConfig(
            target_col="math_score",
            drop_columns=field(
                default_factory = lambda:["reading_score", "writing_score"]
            )
        )
    )

    train_arr,test_arr,preprocessor_path=transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
        run_dir=run_dir
    )

    #model_training

    trainer=ModelTrainer()
    score=trainer.initiate_model_trainer(
        train_arr=train_arr,
        test_arr=test_arr,
        run_dir=run_dir
    )

    logging.info(f"TRAINING PIPELINE COMPLETED | R2 SCORE = {score}")
    print("Training completed. R2 score:", score)


if __name__ == "__main__":
    run_pipeline()