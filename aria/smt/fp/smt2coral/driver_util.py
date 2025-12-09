# vim: set sw=4 ts=4 softtabstop=4 expandtab:
import argparse
import logging

_logger = logging.getLogger(__name__)


def parserAddLoggerArg(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    parser.add_argument("-l", "--log-level", type=str, default="info",
                        dest="log_level",
                        choices=['debug', 'info', 'warning', 'error'])
    parser.add_argument("--log-file",
                        dest='log_file',
                        type=str,
                        default=None,
                        help="Log to specified file")
    parser.add_argument("--log-only-file",
                        dest='log_only_file',
                        action='store_true',
                        default=False,
                        help='Only log to file specified by --log-file and not the console')
    parser.add_argument("--log-show-src-locs",
        dest="log_show_source_locations",
        action='store_true',
        default=False,
        help='Include source locations in log'
    )
    return


def handleLoggerArgs(pargs, parser):
    assert isinstance(pargs, argparse.Namespace)
    assert isinstance(parser, argparse.ArgumentParser)
    logLevel = getattr(logging, pargs.log_level.upper(), None)
    if logLevel == logging.DEBUG:
        logFormat = ('%(levelname)s:%(threadName)s: %(filename)s:%(lineno)d '
                     '%(funcName)s()  : %(message)s')
    else:
        if pargs.log_show_source_locations:
            logFormat = '%(levelname)s:%(threadName)s %(filename)s:%(lineno)d : %(message)s'
        else:
            logFormat = '%(levelname)s:%(threadName)s: %(message)s'

    if not pargs.log_only_file:
        # Add default console level with appropriate formatting and level.
        logging.basicConfig(level=logLevel, format=logFormat)
    else:
        if pargs.log_file is None:
            parser.error('--log-file-only must be used with --log-file')
        logging.getLogger().setLevel(logLevel)
    if pargs.log_file is not None:
        file_handler = logging.FileHandler(pargs.log_file)
        log_formatter = logging.Formatter(logFormat)
        file_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_handler)
