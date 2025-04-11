import logging
import pytest
import sys
import typing

from ktoolbox import common


def test_logger_1() -> None:
    name = "foo.1"
    logger = common.ExtendedLogger(name)
    assert isinstance(logger, logging.Logger)
    assert isinstance(logger, common.ExtendedLogger)
    assert logger.name == name
    assert logger.wrapped_logger is logging.getLogger(name)

    if sys.version_info >= (3, 11):
        typing.assert_type(logger.wrapped_logger, logging.Logger)

    common.log_config_logger(logging.DEBUG, logger)
    assert logger.level == logging.DEBUG
    assert logger.wrapped_logger.level == logging.DEBUG

    common.log_config_logger(logging.INFO, logger, logger.wrapped_logger)
    assert logger.level == logging.INFO
    assert logger.wrapped_logger.level == logging.INFO
    assert logging.getLogger(name).level == logging.INFO

    assert logger.handlers is logger.wrapped_logger.handlers
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], common._LogHandler)

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        logger.error_and_exit("calling-error-and-exit", exit_code=42)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 42

    logger.info(f"test {name}")
