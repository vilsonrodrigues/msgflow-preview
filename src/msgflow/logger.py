import datetime
import logging
import os
import sys
from functools import partial
from logging import Logger
from logging.config import dictConfig
from os import path
from typing import Optional
import msgspec

from msgflow.envs import envs

_FORMAT = (f"{envs.logging_prefix}%(levelname)s %(asctime)s "
           "%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "msgflow": {
            "class": "msgflow.utils.logging.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "msgflow": {
            "class": "logging.StreamHandler",
            "formatter": "msgflow",
            "level": envs.logging_level,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "msgflow": {
            "handlers": ["msgflow"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}

def _configure_msgflow_root_logger() -> None:
    logging_config = dict()
    
    if not envs.configure_logging and envs.logging_config_path:
        raise RuntimeError(
            "MSGFLOW_CONFIGURE_LOGGING evaluated to false, but "
            "MSGFLOW_LOGGING_CONFIG_PATH was given. MSGFLOW_LOGGING_CONFIG_PATH "
            "implies MSGFLOW_CONFIGURE_LOGGING. Please enable "
            "MSGFLOW_CONFIGURE_LOGGING or unset MSGFLOW_LOGGING_CONFIG_PATH.")

    if envs.configure_logging:
        logging_config = DEFAULT_LOGGING_CONFIG

    if envs.logging_config_path:
        if not path.exists(envs.logging_config_path):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                envs.logging_config_path)
        with open(envs.logging_config_path, encoding="utf-8") as f:
            custom_config = msgspec.json.decode(f.read())

        if not isinstance(custom_config, dict):
            raise TypeError("Invalid logging config. Expected Dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    return logger


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_msgflow_root_logger()

logger = init_logger(__name__)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ["call", "return"]:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the msgflow root_dir
            return
        # Log every function call or return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == "call":
                    f.write(f"{ts} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    msgflow root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "MSGFLOW_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # By default, this is the msgflow root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))