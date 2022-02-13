import logging
import sys
from pathlib import Path
from typing import Optional


class LoggingBuilder:

    def __init__(self, logger: logging.Logger, formatter: Optional[logging.Formatter] = None,
                 root_level: int = logging.DEBUG):
        self._logger: logging.Logger = logger
        self._logger.setLevel(root_level)
        if formatter is None:
            formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(name)-15s %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
        self._formatter: logging.Formatter = formatter

    def add_console_handler(self, level: int = logging.DEBUG):
        from logging import StreamHandler
        handler = StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(self._formatter)
        self._logger.addHandler(handler)

    def add_file_rotating_handler(self, logfile: str, max_bytes: int = 50000000, backup_count: int = 5,
                                  level: int = logging.DEBUG):
        logfile = Path(logfile).with_suffix(".log")
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(filename=logfile,
                                      mode="a",
                                      maxBytes=max_bytes,
                                      backupCount=backup_count)
        handler.setLevel(level)
        handler.setFormatter(self._formatter)
        self._logger.addHandler(handler)

    def disable_default_external_loggers(self):
        loggers_to_disable = ["matplotlib",
                              "PIL",
                              "h5py",
                              "urllib3",
                              "uamqp",
                              "azure.core.pipeline.policies.http_logging_policy",
                              "msal.authority",
                              "msal.application",
                              "msal.telemetry",
                              "msal.token_cache",
                              "azure.identity._internal.get_token_mixin"]
        for logger in loggers_to_disable:
            self.disable_logger_by_name(logger)

    def disable_logger_by_name(self, name: str):
        logger_to_disable = logging.getLogger(name)
        logger_to_disable.propagate = False
        logger_to_disable.level = logging.WARNING

    def add_default_loggers(self, logfile: str) -> logging.Logger:
        self.add_console_handler()
        self.add_file_rotating_handler(logfile)
        self.disable_default_external_loggers()
        return self._logger
