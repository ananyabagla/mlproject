import logging
import os
from datetime import datetime

# Create log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

# Configure logging to output to BOTH file and console
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),  # Log to file
        logging.StreamHandler()  # ✅ Log to console (terminal)
    ]
)