# coding: utf-8
"""
SMTLIB solver interface.

Partially modified from https://github.com/trailofbits/manticore
TODO: allow the user to select different modes
1. Use the same process to first accept the whole formula, and then accept
   multiple (check-sat-assuming) commands?
2. Every time, create a new process the solve each individual instance
   (including formulas and check-sat-assuming)

TODO: we need to consider three "incremental mode"
1. Every time build new assertions
   - build new solver process + new assertions?
   - reuse the solver process, but use `reset` command? (it seems that
     `reset` can affect the tactic)
2. Use push/pop
3. Use assumption literal
"""
import fcntl  # seems that Windows does not have this lib.
import logging
import shlex
import time
from subprocess import PIPE, Popen
from typing import Optional, List
import os

from aria.efmc.smttools.smt_exceptions import SolverError

logger = logging.getLogger(__name__)


class SmtlibProc:
    """Single SMTLIB interactive process."""

    def __init__(self, solver_command: str, debug=False):
        """Single smtlib interactive process
        :param solver_command: the shell command to execute
        :param debug: log all messaging
        """
        self._proc: Optional[Popen] = None
        self._command = solver_command
        self._debug = debug
        self._last_buf = ""

    def start(self):
        """Spawns POpen solver process"""
        if self._proc is not None:
            return
        # bufsize=0: if we set input to unbuffered, we get syntax errors
        # in large formulas
        self._proc = Popen(
            shlex.split(self._command),
            stdin=PIPE,
            stdout=PIPE,
            universal_newlines=True,
            close_fds=True,
        )

        # stdout should be non-blocking
        fl = fcntl.fcntl(self._proc.stdout, fcntl.F_GETFL)
        fcntl.fcntl(self._proc.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        self._last_buf = ""

    def stop(self):
        """
        Stops the solver process by:
        - sending a SIGKILL signal,
        - waiting till the process terminates (so we don't leave a zombie process)
        """
        if self._proc is None:
            return
        # if it did not finished already
        if self._proc.returncode is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            # Kill the process
            self._proc.kill()
            self._proc.wait()

            # No need to wait for termination, zombies avoided.
        self._proc = None

    def send(self, cmd: str):
        """
        Send a string to the solver.
        :param cmd: a SMTLIBv2 command (ex. (check-sat))
        """
        if self._debug:
            logger.debug(">%s", cmd)
        assert self._proc is not None
        try:
            self._proc.stdout.flush()  # type: ignore
            self._proc.stdin.write(f"{cmd}\n")  # type: ignore
            self._proc.stdin.flush()  # type: ignore
        except (BrokenPipeError, IOError) as e:
            logger.critical(
                "Solver encountered an error trying to send commands: %s.\n"
                "\tOutput: %s\n\n"
                "\tStderr: %s",
                e,
                self._proc.stdout,
                self._proc.stderr,
            )
            raise

    def recv(self, wait=True):
        """Reads the response from the smtlib solver
        :param wait: a boolean that indicate to wait with a blocking call
        until the results are available. Otherwise, it returns None if the solver
        does not respond.
        """
        tries = 0
        timeout = 0.0

        buf = ""
        if self._last_buf != "":  # we got a partial response last time, let's use it
            buf = buf + self._last_buf

        while True:
            try:
                buf = buf + self._proc.stdout.read()  # type: ignore
                buf = buf.strip()
            except TypeError:
                if not wait:
                    if (
                        buf != ""
                    ):  # we got an error, but something was returned, let's save it
                        self._last_buf = buf
                    return None
                else:
                    tries += 1

            if buf == "":
                continue

            # this verifies if the response from the solver is complete
            # (it has balanced brackets)
            # TODO: the results of some special queries might not be
            # s-expression (e.g., it can be multiple e-expressions)
            lparen, rparen = map(sum, zip(*((c == "(", c == ")") for c in buf)))
            if lparen == rparen and buf != "":
                break

            if tries > 3:
                time.sleep(timeout)
                timeout += 0.1
                # timeout += 0.5

        buf = buf.strip()
        self._last_buf = ""

        if "(error" in buf or "Fatal" in buf:
            raise SolverError(f"Solver error: {buf}")

        if self._debug:
            logger.debug("<%s", buf)

        return buf

    def is_started(self):
        return self._proc is not None

    def clear_buffers(self):
        self._proc.stdout.flush()
        self._proc.stdin.flush()


class SMTLIBSolver:
    """SMTLIB solver interface using external solver via subprocess."""

    def __init__(self, solver_command: str, debug=False):
        """
        Build a smtlib solver instance.
        This is implemented using an external solver (via a subprocess).
        """
        self._smtlib = SmtlibProc(solver_command, debug)

        self._smtlib.start()
        # run solver specific initializations

    def check_sat(self):
        """
        check sat
        """
        start = time.time()
        self._smtlib.send(f"(check-sat)")
        status = self._smtlib.recv()  # is this correct?
        assert status is not None

        logger.debug("Check took %s seconds (%s)", time.time() - start, status)

        if status in ("sat", "unsat", "unknown"):
            return status
        raise SolverError(status)
        # return False

    def check_sat_assuming(self, assumptions: List[str]):
        """
        :param assumptions: a list of assumption literal
        FIXME: implement and test; figure out what should "assumptions" look like
        """
        start = time.time()
        all_expressions_str = " ".join(assumptions)
        self._smtlib.send(f"(check-sat-assuming ({all_expressions_str}))")
        status = self._smtlib.recv()  # is this correct?
        assert status is not None

        logger.debug("Check took %s seconds (%s)", time.time() - start, status)

        if status in ("sat", "unsat", "unknown"):
            return status
        raise SolverError(status)
        # return False

    def check_sat_with_pushpop_scope(self, new_assertions: str):
        """
        :param new_assertions: is a tmp cnt
        FIXME: implement and test
        """
        start = time.time()
        self.push()
        self._smtlib.send(new_assertions)
        self._smtlib.send("(check-sat)")
        status = self._smtlib.recv()  # is this correct?
        self.push()
        assert status is not None

        logger.debug("Check took %s seconds (%s)", time.time() - start, status)

        if status in ("sat", "unsat", "unknown"):
            return status
        raise SolverError(status)
        # return False

    def check_sat_from_scratch(self, whole_fml: str):
        """
        Check the satisfiability of the current state
        :param whole_fml: should be a whole formula (with declaration,
                         assertions, and check-sat)
        :return: whether current state is satisfiable or not.
        """
        start = time.time()
        self._smtlib.send(whole_fml)
        status = self._smtlib.recv()
        assert status is not None

        logger.debug("Check took %s seconds (%s)", time.time() - start, status)

        if status in ("sat", "unsat", "unknown"):
            return status
        raise SolverError(status)
        # return False

    def assert_assertions(self, assertions: str):
        """
        Add new assertions
        """
        self._smtlib.send(assertions)

    # push pop
    def push(self):
        """Pushes and save the current constraint store and state."""
        self._smtlib.send("(push 1)")

    def pop(self):
        """Recall the last pushed constraint store and state."""
        self._smtlib.send("(pop 1)")

    def get_expr_values(self, expressions: List[str]):
        """
        If satisfiable, fetch the values of expressions
        """
        all_expressions_str = " ".join(expressions)
        self._smtlib.send(f"(get-value ({all_expressions_str}))")
        ret_solver = self._smtlib.recv()
        assert ret_solver is not None
        return ret_solver  # return raw string

    def get_unsat_core(self):
        """
        Get the unsat core.
        FIXME: implement and test
        """
        cmd = "(get-unsat-core)"  # core or cores?
        self._smtlib.send(cmd)
        ret = self._smtlib.recv()
        return ret

    def reset(self):
        """
        Auxiliary method to reset the smtlib external solver to initial defaults
        TODO: Z3 and CVC5 supports the "reset" cmd. Maybe can use it
        """
        self._smtlib.stop()  # does not do anything if already stopped
        self._smtlib.start()
        self._smtlib.clear_buffers()  # need this?

    def stop(self):
        """Stop the solver process."""
        self._smtlib.stop()
