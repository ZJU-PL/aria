"""Exceptions for efbv parallel module."""

from aria.utils.exceptions import SMTSuccess, SMTUnknown


class ExitsSolverSuccess(SMTSuccess):
    """The Exists Solver computes a candidate."""


class ForAllSolverSuccess(SMTSuccess):
    """The Forall Solver validates the candidate as feasible(?)."""


class ExitsSolverUnknown(SMTUnknown):
    """TBD."""


class ForAllSolverUnknown(SMTUnknown):
    """TBD."""
