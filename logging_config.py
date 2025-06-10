import logging
from pathlib import Path


def setup_logging(module_name=None):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / 'project.log'

    logger = logging.getLogger(module_name or __name__)
    logger.setLevel(logging.INFO)

    # Очищаем существующие обработчики
    if logger.hasHandlers():
        logger.handlers.clear()

    # Форматер
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Файловый обработчик
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
