import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    dataset_path: Path = Path("notebook/data/stud.csv")

    test_size: float = 0.2
    random_state: int = 42

    target_col: str = "math_score"
    stratify: bool = False   # regression → False


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def _create_run_dir(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.project_root / "artifacts" / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Run directory created at {run_dir}")
        return run_dir

    def _validate_dataset(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Dataset is empty")

        if self.config.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            dataset_path = self.config.project_root / self.config.dataset_path
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            df = pd.read_csv(dataset_path)
            logging.info(f"Dataset loaded with shape {df.shape}")

            self._validate_dataset(df)

            run_dir = self._create_run_dir()

            # save raw
            raw_path = run_dir / "raw.csv"
            df.to_csv(raw_path, index=False)

            stratify_col = df[self.config.target_col] if self.config.stratify else None

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_col
            )

            train_path = run_dir / "train.csv"
            test_path = run_dir / "test.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info("Data ingestion completed successfully")

            return train_path, test_path, run_dir

        except Exception as e:
            logging.esxception("Data transformation failed")
            print("ACTUAL ERROR:", e)
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()
