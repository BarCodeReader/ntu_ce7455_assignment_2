import logging


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    formatter = (
        "%(asctime)s - {color}%(levelname)s{reset} - %(filename)s:%(lineno)d"
        " - %(module)s.%(funcName)s - %(process)d - %(message)s"
    )
    FORMATS = {
        logging.DEBUG: formatter.format(color=grey, reset=reset),
        logging.INFO: formatter.format(color=green, reset=reset),
        logging.WARNING: formatter.format(color=yellow, reset=reset),
        logging.ERROR: formatter.format(color=red, reset=reset),
        logging.CRITICAL: formatter.format(color=bold_red, reset=reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def register_logger(logging_path):
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logging_path)
    file_handler.setFormatter(
        logging.Formatter(
            (
                "%(asctime)s - %(filename)s:%(lineno)d"
                " - %(module)s.%(funcName)s - %(process)d - %(message)s"
            ),
            "%Y-%m-%dT%H:%M:%S.%f%z",
        )
    )
    logger.addHandler(file_handler)

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(ColorfulFormatter())
    logger.addHandler(streaming_handler)

    return logger
