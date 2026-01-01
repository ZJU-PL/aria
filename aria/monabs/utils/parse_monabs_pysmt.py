"""
PySMT-based parser for monadic predicate abstraction inputs.

The expected SMT-LIB structure matches the Z3 parser in `parse_monabs.py`:
- Assertions before the first `push` form the shared precondition.
- Constraints between a `push` and the matching `pop` form one predicate
  block and are collected into `cnt_list`.
"""

from typing import List, Tuple

from pysmt.shortcuts import And, TRUE
from pysmt.smtlib.parser import SmtLibParser


class MonAbsPySMTParser:  # pylint: disable=too-few-public-methods
    """Lightweight parser that extracts (precond, cnt_list) using PySMT."""

    def __init__(self) -> None:
        self.precond = TRUE()
        self.cnt_list: List = []
        self._current_cnt = TRUE()
        self._in_cnt = False

    def _consume_command(self, cmd) -> None:
        """Process a single SMT-LIB command from the parsed script."""
        name = cmd.name
        if name == "assert":
            fml = cmd.args[0]
            if self._in_cnt:
                self._current_cnt = And(self._current_cnt, fml)
            else:
                self.precond = And(self.precond, fml)
        elif name == "push":
            # Start collecting a new constraint block.
            self._in_cnt = True
            self._current_cnt = TRUE()
        elif name == "pop":
            # Finish the current constraint block.
            if self._in_cnt:
                self.cnt_list.append(self._current_cnt)
            self._current_cnt = TRUE()
            self._in_cnt = False
        elif name == "check-sat":
            # Oracle calls are ignored for parsing purposes.
            return
        else:
            # Declarations and set-logic are handled by PySMT during parsing.
            return

    def parse_file(self, filename: str) -> Tuple:
        """Parse an SMT-LIB file and return (precond, cnt_list)."""
        parser = SmtLibParser()
        script = parser.get_script_fname(filename)
        for cmd in script.commands:
            self._consume_command(cmd)

        # If the file ends inside a push scope, keep the collected block.
        if self._in_cnt:
            self.cnt_list.append(self._current_cnt)
            self._current_cnt = TRUE()
            self._in_cnt = False

        return self.precond, self.cnt_list


def parse_monabs_pysmt(filename: str) -> Tuple:
    """Convenience wrapper returning (precond, cnt_list)."""
    parser = MonAbsPySMTParser()
    return parser.parse_file(filename)
