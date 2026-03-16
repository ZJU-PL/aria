from aria.smt.pcdclt import solve as pcdclt_solve
from aria.smt.pcdclt import config as pcdclt_config
import sys
import os
import signal
import logging
import logging.config

class TimerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "is_timing", False) is True

if __name__ == "__main__":
    """
    TODO: move it to aria.cli after it's OK?
    """
    logging.basicConfig(level=logging.INFO)
    log_dir = "/tmp/pcdclt"
    os.makedirs(log_dir, exist_ok=True)
    timer_log_file = os.path.join(log_dir, "timer.log")
    normal_log_file = os.path.join(log_dir, "normal.log")
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
            },
            "timer": {
                "format": "%(asctime)s %(name)s %(message)s"
            },
        },

        "filters": {
            "timer_only": {
                "()": TimerFilter,
            },
        },

        "handlers": {
            "app_file": {
                "class": "logging.FileHandler",
                "filename": normal_log_file,
                "formatter": "default",
                "level": "INFO",
            },
            "timer_file": {
                "class": "logging.FileHandler",
                "filename": timer_log_file,
                "formatter": "timer",
                "filters": ["timer_only"],
                "level": "INFO",
            },
        },

        "root": {
            "handlers": ["app_file", "timer_file"],
            "level": "INFO",
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)

    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(SystemExit(124)))

    input_file = sys.argv[1]

    with open(input_file, 'r', encoding='utf-8') as f:
        smt2_string = f.read()

    logic = 'ALL'
    for line in smt2_string.split('\n'):
        if line.strip().startswith('(set-logic'):
            logic = line.split()[1].rstrip(')')
            break

    result = pcdclt_solve(smt2_string, logic=logic)
    print(result.name.lower())
