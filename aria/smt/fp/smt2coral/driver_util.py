# vim: set sw=4 ts=4 softtabstop=4 expandtab:
import argparse
import logging

_logger = logging.getLogger(__name__)


def parser_add_logger_arg(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="info",
        dest="log_level",
        choices=["debug", "info", "warning", "error"],
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        type=str,
        default=None,
        help="Log to specified file",
    )
    parser.add_argument(
        "--log-only-file",
        dest="log_only_file",
        action="store_true",
        default=False,
        help="Only log to file specified by --log-file " "and not the console",
    )
    parser.add_argument(
        "--log-show-src-locs",
        dest="log_show_source_locations",
        action="store_true",
        default=False,
        help="Include source locations in log",
    )


# Backward compatibility
parserAddLoggerArg = parser_add_logger_arg


def handle_logger_args(pargs, parser):
    assert isinstance(pargs, argparse.Namespace)
    assert isinstance(parser, argparse.ArgumentParser)
    log_level = getattr(logging, pargs.log_level.upper(), None)
    if log_level == logging.DEBUG:
        log_format = (
            "%(levelname)s:%(threadName)s: %(filename)s:%(lineno)d "
            "%(funcName)s()  : %(message)s"
        )
    else:
        if pargs.log_show_source_locations:
            log_format = (
                "%(levelname)s:%(threadName)s " "%(filename)s:%(lineno)d : %(message)s"
            )
        else:
            log_format = "%(levelname)s:%(threadName)s: %(message)s"

    if not pargs.log_only_file:
        # Add default console level with appropriate formatting and level.
        logging.basicConfig(level=log_level, format=log_format)
    else:
        if pargs.log_file is None:
            parser.error("--log-file-only must be used with --log-file")
        logging.getLogger().setLevel(log_level)
    if pargs.log_file is not None:
        file_handler = logging.FileHandler(pargs.log_file)
        log_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_handler)


# Backward compatibility
handleLoggerArgs = handle_logger_args
