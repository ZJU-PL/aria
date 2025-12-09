from aria.utils.exceptions import SMTSuccess, SMTError, SMTUnknown


class ExitsSolverSuccess(SMTSuccess):
    """The Exists Solver computes a candidate"""
    pass


class ForAllSolverSuccess(SMTSuccess):
    """The Forall Solver validates the candidate as feasible(?)"""
    pass


class ExitsSolverUnknown(SMTUnknown):
    """TBD"""
    pass


class ForAllSolverUnknown(SMTUnknown):
    """TBD
    """
    pass
