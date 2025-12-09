import os
import shutil
from typing import Any

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class Theorem:
    Farkas = 'farkas'
    Handelman = 'handelman'
    Putinar = 'putinar'


class AvailabilityDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if not isinstance(value, list):
                self[key] = [value]

    def __getitem__(self, key: Any) -> Any:
        possible_values = super().__getitem__(key)
        for value in possible_values:
            if AvailabilityDict.available(value):
                return value
        return None

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(value, list):
            value = [value]
        super().__setitem__(key, value)

    @staticmethod
    def available(item):
        return shutil.which(item) is not None


class Constant:
    """This class consist of some constant dictionaries which are used for configuration of the solvers.

    """
    options = {
        'z3': '(set-option :print-success false)\n' +
              '(set-option :produce-models true)\n',
        'mathsat': '(set-option :print-success false)\n' +
                   '(set-option :produce-models true)\n',
        'default': ''

    }

    default_path = AvailabilityDict({
        'z3': ['z3', os.path.join(ABS_PATH, '..', '..', 'solver', 'z3')],
        'mathsat': ['mathsat', os.path.join(ABS_PATH, '..', '..', 'solver', 'mathsat')],
        'default': ''
    })

    command = {
        'z3': '',
        'mathsat': '',
        'default': ''
    }
