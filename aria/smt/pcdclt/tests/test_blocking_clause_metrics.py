from importlib import import_module

from aria.tests import TestCase, main
from aria.utils import SolverResult


def _solver_contract():
    solver = import_module("aria.smt.pcdclt.solver")
    return (
        getattr(solver, "BlockingClauseMetrics"),
        getattr(solver, "new_blocking_clause_metrics"),
        getattr(solver, "record_added_clauses"),
        getattr(solver, "record_theory_result"),
    )


class TestBlockingClauseMetrics(TestCase):
    def test_unsat_generates_one_clause_before_simplification(self):
        metric_type, new_metrics, _, record_result = _solver_contract()

        metrics = new_metrics()

        clause = record_result(
            metrics,
            SolverResult.UNSAT,
            [1, -2, 3],
        )

        self.assertIsInstance(metrics, metric_type)
        self.assertEqual(clause, [-1, 2, -3])
        self.assertEqual(metrics.generated_total, 1)
        self.assertEqual(metrics.theory_unsat_checks_total, 1)
        self.assertEqual(metrics.theory_sat_checks_total, 0)
        self.assertEqual(metrics.theory_unknown_or_error_total, 0)
        self.assertEqual(metrics.added_total, 0)

    def test_sat_does_not_increment_generated_total(self):
        _, new_metrics, _, record_result = _solver_contract()

        metrics = new_metrics()

        clause = record_result(metrics, SolverResult.SAT, [1, -2])

        self.assertIsNone(clause)
        self.assertEqual(metrics.generated_total, 0)
        self.assertEqual(metrics.theory_sat_checks_total, 1)
        self.assertEqual(metrics.theory_unsat_checks_total, 0)
        self.assertEqual(metrics.theory_unknown_or_error_total, 0)

    def test_unknown_does_not_increment_generated_total(self):
        _, new_metrics, _, record_result = _solver_contract()

        metrics = new_metrics()

        clause = record_result(metrics, SolverResult.UNKNOWN, [1])

        self.assertIsNone(clause)
        self.assertEqual(metrics.generated_total, 0)
        self.assertEqual(metrics.theory_sat_checks_total, 0)
        self.assertEqual(metrics.theory_unsat_checks_total, 0)
        self.assertEqual(metrics.theory_unknown_or_error_total, 1)

    def test_worker_error_does_not_increment_generated_total(self):
        _, new_metrics, _, record_result = _solver_contract()

        metrics = new_metrics()

        clause = record_result(metrics, "ERROR:worker died", [1, 2])

        self.assertIsNone(clause)
        self.assertEqual(metrics.generated_total, 0)
        self.assertEqual(metrics.theory_sat_checks_total, 0)
        self.assertEqual(metrics.theory_unsat_checks_total, 0)
        self.assertEqual(metrics.theory_unknown_or_error_total, 1)

    def test_added_total_is_tracked_after_simplification(self):
        _, new_metrics, record_added, record_result = _solver_contract()

        metrics = new_metrics()

        first = record_result(metrics, SolverResult.UNSAT, [1, 2])
        second = record_result(metrics, SolverResult.UNSAT, [1, -2])
        self.assertEqual(metrics.generated_total, 2)

        simplified_clauses = [[-1]]
        record_added(metrics, simplified_clauses)

        self.assertEqual(first, [-1, -2])
        self.assertEqual(second, [-1, 2])
        self.assertEqual(metrics.generated_total, 2)
        self.assertEqual(metrics.added_total, 1)


if __name__ == "__main__":
    main()
