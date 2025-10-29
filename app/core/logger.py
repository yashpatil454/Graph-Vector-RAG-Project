import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

class SingletonLogger:
    _instance = None
    _lock = Lock()  # Thread safety

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
                    cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"

        # Create and configure logger
        self.logger = logging.getLogger("FastAPIAppLogger")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if re-imported
        if not self.logger.handlers:
            handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
            formatter = logging.Formatter(
                "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
