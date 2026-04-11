import os
import tempfile
from unittest import mock

import z3

from aria.tests import TestCase, main
from aria.quant.qe import qe_lme_parallel


class FakeFuture:
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._value


class FakeExecutor:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.submit_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        self.submit_calls.append((fn, args, kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            return FakeFuture(error=outcome)
        return FakeFuture(value=outcome)


class FakeNamedTemporaryFile:
    def __init__(self, path):
        self.name = path
        self._file = open(path, "w+", encoding="utf-8")

    def write(self, content):
        return self._file.write(content)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._file.close()
        return False


class TestParallelQuantifierElimination(TestCase):
    def test_missing_solver_path_returns_false(self):
        x = z3.Int("x")

        with mock.patch.object(qe_lme_parallel, "resolve_z3_path", return_value=None):
            with mock.patch.object(qe_lme_parallel, "extract_models") as mock_extract:
                result = qe_lme_parallel.qelim_exists_lme_parallel(
                    x > 0,
                    [x],
                    max_iterations=1,
                )

        self.assertEqual(result, "false")
        mock_extract.assert_not_called()

    def test_final_result_deduplicates_duplicate_projections(self):
        x = z3.Int("x")
        fake_executor = FakeExecutor(["(= y 1)", "(= y 1)"])
        models = [
            {"y": {"type": "Int", "value": "1"}},
            {"y": {"type": "Int", "value": "1"}},
        ]

        with mock.patch.object(qe_lme_parallel, "resolve_z3_path", return_value="/fake/z3"):
            with mock.patch.object(qe_lme_parallel, "extract_models", return_value=models):
                with mock.patch.object(
                    qe_lme_parallel, "ProcessPoolExecutor", return_value=fake_executor
                ):
                    with mock.patch.object(
                        qe_lme_parallel, "as_completed", side_effect=lambda futures: futures
                    ):
                        result = qe_lme_parallel.qelim_exists_lme_parallel(
                            x > 0,
                            [x],
                            num_workers=2,
                            batch_size=2,
                            max_iterations=1,
                        )

        self.assertEqual(result, "(= y 1)")

    def test_explicit_bounds_and_timeout_are_used(self):
        x = z3.Int("x")
        fake_executor = FakeExecutor(["(= y 3)"])
        models = [{"y": {"type": "Int", "value": "3"}}]

        with mock.patch.object(qe_lme_parallel, "resolve_z3_path", return_value="/fake/z3"):
            with mock.patch.object(qe_lme_parallel, "extract_models", return_value=models) as mock_extract:
                with mock.patch.object(
                    qe_lme_parallel, "ProcessPoolExecutor", return_value=fake_executor
                ):
                    with mock.patch.object(
                        qe_lme_parallel, "as_completed", side_effect=lambda futures: futures
                    ):
                        result = qe_lme_parallel.qelim_exists_lme_parallel(
                            x > 0,
                            [x],
                            num_workers=1,
                            batch_size=1,
                            max_iterations=1,
                            solver_timeout=7,
                        )

        self.assertEqual(result, "(= y 3)")
        self.assertEqual(mock_extract.call_count, 1)
        self.assertEqual(mock_extract.call_args.kwargs["solver_timeout"], 7)
        self.assertEqual(fake_executor.submit_calls[0][1][2], 7)
        self.assertEqual(fake_executor.submit_calls[0][1][3], "/fake/z3")

    def test_worker_failure_does_not_discard_other_projections(self):
        x = z3.Int("x")
        fake_executor = FakeExecutor([RuntimeError("worker failed"), "(= y 2)"])
        models = [
            {"y": {"type": "Int", "value": "1"}},
            {"y": {"type": "Int", "value": "2"}},
        ]

        with mock.patch.object(qe_lme_parallel, "resolve_z3_path", return_value="/fake/z3"):
            with mock.patch.object(qe_lme_parallel, "extract_models", return_value=models):
                with mock.patch.object(
                    qe_lme_parallel, "ProcessPoolExecutor", return_value=fake_executor
                ):
                    with mock.patch.object(
                        qe_lme_parallel, "as_completed", side_effect=lambda futures: futures
                    ):
                        result = qe_lme_parallel.qelim_exists_lme_parallel(
                            x > 0,
                            [x],
                            num_workers=2,
                            batch_size=2,
                            max_iterations=1,
                        )

        self.assertEqual(result, "(= y 2)")

    def test_run_z3_script_cleans_temp_file_after_subprocess_failure(self):
        fd, temp_path = tempfile.mkstemp(suffix=".smt2")
        os.close(fd)
        os.unlink(temp_path)

        with mock.patch.object(qe_lme_parallel, "resolve_z3_path", return_value="/fake/z3"):
            with mock.patch.object(
                qe_lme_parallel.tempfile,
                "NamedTemporaryFile",
                return_value=FakeNamedTemporaryFile(temp_path),
            ):
                with mock.patch.object(
                    qe_lme_parallel.subprocess,
                    "run",
                    side_effect=OSError("subprocess failed"),
                ):
                    result = qe_lme_parallel.run_z3_script(
                        "(check-sat)",
                        timeout=1,
                    )

        self.assertIsNone(result)
        self.assertFalse(os.path.exists(temp_path))


if __name__ == "__main__":
    main()
