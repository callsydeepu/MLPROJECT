from pathlib import Path
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = logs_dir / LOG_FILE

logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)



if __name__=="__main__":
    logging.info("logging has started")


