"""
Validate the unsat core returned by the cvc5_minimizer.
"""

try:
    import cvc5  # pylint: disable=import-error
except ImportError:
    cvc5 = None  # type: ignore

from aria.ml.llm.unsat_core.cvc5_minimizer import (
    build_smt2_formula_from_string_constraints,
    parse_input_formula,
    set_cvc5_options_for_unsat,
    set_cvc5_options_for_mus,
)


class UNSATVerifier:
    """A verifier class which checks for the unsatisfiability of a given set
    of constraints.

    Parameters:
        solver (cvc5.Solver): An SMT2 solver instance.
        statistics (dict): Statistics for solving the given set of constraints.
    """

    def __init__(self):
        """Initialize :class: ``intellismt.modules.verifiers.UNSATVerifier``."""
        self.solver = cvc5.Solver()
        set_cvc5_options_for_unsat(self.solver)
        self.statistics = None

    def reset(self):
        """Reset all assertions added to the solver, helps with incremental solving."""
        self.solver.resetAssertions()

    def check(self, constraints, all_constraints, all_smt2_constraints, placeholder):
        """Checks whether the given set of ``constraints`` is unsatisfiable.

        Arguments:
            constraints (list): Constraint subset from GPT-x's response in string format.
            all_constraints (list): Complete list of constraints in string format.
            all_smt2_constraints (list): Complete list of constraints in SMT2-Lib
                format. There is one-to-one correspondence with constraints in
                ``all_constraints``.
            placeholder (str): SMT2-Lib input file string placeholder, where all
                assertions are represented by "<ASSERT>" keyword.

        Returns:
            (bool): ``True``, if unsatisfiable, else ``False``.
        """
        smt2_formula = build_smt2_formula_from_string_constraints(
            constraints, all_constraints, all_smt2_constraints, placeholder
        )
        if "set-logic" not in smt2_formula:
            try:
                self.solver.setLogic("QF_SLIA")
            except Exception:
                pass

        parse_input_formula(self.solver, smt2_formula, "smt_formula")

        result = self.solver.checkSat().isUnsat()
        statistics_dict = self.solver.getStatistics().get()
        setattr(self, "statistics", statistics_dict)

        if result:
            return True
        self.reset()
        return False


class MUSVerifier:
    """A verifier class which checks whether a given set of constraints is a
    minimally unsatisfiable subset (MUS). For a set of constraints to be minimally
    unsatisfiable, upon removing any of the constraints within it, it should
    become satisfiable.

    Parameters:
        solver (cvc5.Solver): An SMT2 solver instance.
        statistics (dict): Statistics for solving the given set of constraints.
    """

    def __init__(self):
        """Initialize :class: ``intellismt.modules.verifiers.MUSVerifier``."""
        self.solver = cvc5.Solver()
        set_cvc5_options_for_mus(self.solver, True)
        self.statistics = None

    def reset(self):
        """Reset all assertions added to the solver, helps with incremental solving."""
        self.solver.resetAssertions()

    def check(self, constraints, all_constraints, all_smt2_constraints, placeholder):
        """Checks whether the given set of ``constraints`` is minimally unsatisfiable.

        Arguments:
            constraints (list): Constraint subset from GPT-x's response in string format.
            all_constraints (list): Complete list of constraints in string format.
            all_smt2_constraints (list): Complete list of constraints in SMT2-Lib
                format. There is one-to-one correspondence with constraints in
                ``all_constraints``.
            placeholder (str): SMT2-Lib input file string placeholder, where all
                assertions are represented by "<ASSERT>" keyword.

        Returns:
            (bool): ``True``, if MUS, else ``False``.
        """
        for idx in range(len(constraints)):
            subset = constraints[:idx] + constraints[idx + 1 :]
            assert len(constraints) == len(subset) + 1

            smt2_formula = build_smt2_formula_from_string_constraints(
                subset, all_constraints, all_smt2_constraints, placeholder
            )

            if "set-logic" not in smt2_formula:
                try:
                    self.solver.setLogic("QF_SLIA")
                except Exception:
                    pass

            parse_input_formula(self.solver, smt2_formula, "smt_formula")
            result = self.solver.checkSat().isSat()

            if result:
                self.reset()
                continue
            return False
        return True
