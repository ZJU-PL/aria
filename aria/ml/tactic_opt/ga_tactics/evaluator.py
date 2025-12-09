#!/usr/bin/env python3
"""Tactic evaluation system supporting Python API and binary Z3 modes."""

import os
import subprocess
import tempfile
import time
import z3


class EvaluationMode:
    """Evaluation modes for tactic sequences."""
    PYTHON_API = "python_api"
    BINARY_Z3 = "binary_z3"


def get_evaluation_mode():
    """Get evaluation mode from environment (default: Python API)."""
    return os.environ.get("Z3_EVALUATION_MODE", EvaluationMode.PYTHON_API)


def get_z3_binary_path():
    """Find Z3 binary in common locations."""
    paths = ["z3", "/usr/bin/z3", "/usr/local/bin/z3", "/opt/local/bin/z3",
             "../bin_solvers/z3", "../../bin_solvers/z3"]

    for path in paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        try:
            subprocess.run([path, "--version"], capture_output=True, check=True)
            return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def pretty_print_tactic(tactic):
    """Print Z3 tactic applied to a test formula."""
    x, y = z3.Int('x'), z3.Int('y')
    goal = z3.Goal()
    goal.add(z3.And(x > 0, y > 0, x + y == 10))
    result = tactic(goal)

    print("Z3 tactic applied to test formula (x > 0 ∧ y > 0 ∧ x + y == 10):")
    for i, subgoal in enumerate(result):
        print(f"Subgoal {i+1}:")
        for j, formula in enumerate(subgoal):
            print(f"  {j+1}. {formula}")


class TacticEvaluator:
    """Evaluates tactic sequences using Python API or binary Z3."""
    PENALTY = 4294967295  # Large penalty for failures

    def __init__(self, mode=None):
        self.mode = mode or get_evaluation_mode()
        self.z3_binary_path = get_z3_binary_path() if self.mode == EvaluationMode.BINARY_Z3 else None
        if self.mode == EvaluationMode.BINARY_Z3 and not self.z3_binary_path:
            raise RuntimeError("Z3 binary not found. Ensure Z3 is installed and in PATH.")

    def evaluate_sequence(self, tactic_seq, smtlib_file=None, timeout=8):
        """Evaluate tactic sequence, return execution time or penalty on failure."""
        if self.mode == EvaluationMode.PYTHON_API:
            return self._evaluate_python_api(tactic_seq, smtlib_file, timeout)
        if self.mode == EvaluationMode.BINARY_Z3:
            return self._evaluate_binary_z3(tactic_seq, smtlib_file, timeout)
        raise ValueError(f"Unknown evaluation mode: {self.mode}")

    def _run_z3_timed(self, cmd, temp_file):
        """Run Z3 command and return timing or penalty."""
        try:
            start = time.time()
            ret = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return time.time() - start if ret == 0 else self.PENALTY
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _evaluate_python_api(self, tactic_seq, smtlib_file=None, timeout=8):
        """Evaluate using Z3 Python API."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tactics', delete=False) as f:
            f.write(tactic_seq.to_string())
            tactics_file = f.name
        return self._run_z3_timed(f"timeout {timeout}s ./run-tests *".split(), tactics_file)

    def _evaluate_binary_z3(self, tactic_seq, smtlib_file=None, timeout=8):
        """Evaluate using binary Z3 executable."""
        if not smtlib_file or not os.path.exists(smtlib_file):
            return self._evaluate_simple_formula(tactic_seq, timeout)

        with open(smtlib_file, 'r') as f:
            content = self._replace_check_sat_with_apply(f.read(), tactic_seq)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(content)
            temp_file = f.name

        return self._run_z3_timed([self.z3_binary_path, temp_file], temp_file)

    def _evaluate_simple_formula(self, tactic_seq, timeout=8):
        """Evaluate on simple test formula."""
        content = "(set-logic QF_LIA)\n(declare-const x Int)\n(declare-const y Int)\n"
        content += "(assert (and (> x 0) (> y 0) (= (+ x y) 10)))\n"
        content += tactic_seq.to_smtlib_apply() + "\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(content)
            temp_file = f.name

        return self._run_z3_timed([self.z3_binary_path, temp_file], temp_file)

    def _replace_check_sat_with_apply(self, smtlib_content, tactic_seq):
        """Replace first (check-sat) with tactic apply."""
        lines, check_sat_found = [], False
        for line in smtlib_content.split('\n'):
            if line.strip() == '(check-sat)' and not check_sat_found:
                lines.append(tactic_seq.to_smtlib_apply())
                check_sat_found = True
            else:
                lines.append(line)
        return '\n'.join(lines)


def run_tests(tactic_seq=None, mode=None, smtlib_file=None, timeout=8):
    """Run tests and return execution time or penalty."""
    from .models import TacticSeq as TacticSeqModel
    return TacticEvaluator(mode).evaluate_sequence(
        tactic_seq or TacticSeqModel(), smtlib_file, timeout
    )


def evaluate_tactic_fitness(tactic_seq, test_files=None, mode=None, timeout=8):
    """Evaluate fitness across multiple test files, return average time."""
    if not test_files:
        for dir_path in ["benchmarks/smtlib2", "benchmarks"]:
            if os.path.exists(dir_path):
                test_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)[:5]
                             if f.endswith('.smt2')]
                if test_files:
                    break

    if not test_files:
        return run_tests(tactic_seq, mode, timeout=timeout)

    total_time, successful_runs = 0.0, 0
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                exec_time = run_tests(tactic_seq, mode, test_file, timeout)
                if exec_time < TacticEvaluator.PENALTY:
                    total_time += exec_time
                    successful_runs += 1
            except Exception:
                continue

    return TacticEvaluator.PENALTY if successful_runs == 0 else total_time / successful_runs
