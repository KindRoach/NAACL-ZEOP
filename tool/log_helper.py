import logging

import tqdm

from tool.path_helper import ROOT_DIR, mkdir_parent

# create formatter
# https://docs.python.org/3/library/logging.html#logrecord-attributes
FORMATTER = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)
        self.flush()


def get_logger(name: str = "logger"):
    new_logger = logging.getLogger(name)
    new_logger.setLevel(logging.DEBUG)

    # create tqdm handler with a higher log level
    ch = TqdmLoggingHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(FORMATTER)

    # add the handlers to the logger if logger is newly created.
    if not new_logger.handlers:
        # logger.addHandler(fh)
        new_logger.addHandler(ch)

    # stop sending log to parent
    new_logger.propagate = False
    return new_logger


def add_log_file(logger, file_name: str):
    # create file handler which logs even debug messages
    path = ROOT_DIR.joinpath(f"out/log/{file_name}")
    mkdir_parent(path)
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(FORMATTER)
    logger.addHandler(fh)


def remove_log_file(logger):
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]


def remove_log_console(logger):
    logger.handlers = [h for h in logger.handlers if not isinstance(h, TqdmLoggingHandler)]


logger = get_logger(ROOT_DIR.name)


def main():
    logger.info("Info message.")
    add_log_file(logger, "test")
    logger.debug("Debug message.")
    logger.warning("Warning message.")
    remove_log_file(logger)
    logger.critical("Critical message.")


if __name__ == '__main__':
    main()
