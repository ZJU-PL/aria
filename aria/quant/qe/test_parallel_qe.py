"""Test module for parallel quantifier elimination."""
import tempfile
import subprocess
import os
import time
import re
from aria.quant.qe.qe_lme_parallel import (
    qelim_exists_lme_parallel, get_declarations, to_smtlib,
    QE_TEMPLATE, Z3_PATH, parse_qe_result
)


def try_direct_qe_by_z3_binary(phi, qvars):
    """Try direct quantifier elimination using Z3 binary"""
    try:
        declarations = get_declarations(phi)
        qvars_smtlib = " ".join([f"({str(var)} {var.sort().sexpr()})" for var in qvars])

        smt_script = QE_TEMPLATE.format(
            declarations=declarations,
            qvars=qvars_smtlib,
            formula=to_smtlib(phi)
        )

        with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(smt_script)

        result = subprocess.run(
            [Z3_PATH, temp_file_path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False
        )
        os.unlink(temp_file_path)

        if "unknown" not in result.stdout.lower() and "error" not in result.stdout.lower():
            output = result.stdout.strip()


            # Parse Z3 goals output
            if "(goals" in output and "(goal" in output:
                goal_match = re.search(
                    r'\(goal\s+(.*?)\s+:precision', output, re.DOTALL)
                if goal_match:
                    goal_content = goal_match.group(1).strip()
                    constraints = [
                        line.strip() for line in goal_content.split('\n')
                        if line.strip() and not line.startswith(':') and line != ')'
                    ]

                    if len(constraints) == 0:
                        return "true"
                    if len(constraints) == 1:
                        return constraints[0]
                    return f"(and {' '.join(constraints)})"

            # Fallback parsing
            if "true" in output.lower():
                return "true"
            if "false" in output.lower():
                return "false"

            return parse_qe_result(result.stdout)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError,
            ValueError, KeyError) as e:
        print(f"Error in Z3 binary QE: {e}")
    return None


def run_single_test(name, formula, qvars):
    """Run differential test on a single formula"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Formula: {formula}")
    print(f"Quantified vars: {[str(v) for v in qvars]}")

    # Method 1: Parallel QE
    start_time = time.time()
    try:
        result1 = qelim_exists_lme_parallel(formula, qvars)
        time1 = time.time() - start_time
    except (ValueError, KeyError, AttributeError, TypeError,
            subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
        result1, time1 = f"ERROR: {e}", time.time() - start_time

    # Method 2: Z3 Binary QE
    start_time = time.time()
    try:
        result2 = try_direct_qe_by_z3_binary(formula, qvars)
        time2 = time.time() - start_time
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError,
            ValueError, KeyError) as e:
        result2, time2 = f"ERROR: {e}", time.time() - start_time

    result1_str = (result1[:80] + '...' if len(str(result1)) > 80 else str(result1))
    result2_str = (result2[:80] + '...' if result2 and len(str(result2)) > 80 else str(result2))
    print(f"Parallel QE:  {result1_str} ({time1:.3f}s)")
    print(f"Z3 Binary:    {result2_str} ({time2:.3f}s)")
    print(f"Match: {'✓' if result1 == result2 else '✗'}")

    if time1 > 0 and time2 > 0:
        speedup = time1 / time2
        print(f"Speedup: {speedup:.1f}x (Z3 binary {'faster' if speedup > 1 else 'slower'})")

    return {
        'name': name,
        'result1': result1,
        'result2': result2,
        'time1': time1,
        'time2': time2,
        'match': result1 == result2
    }


def differential_test():
    """Run comprehensive differential testing"""
    import z3  # pylint: disable=import-outside-toplevel
    print("="*80)
    print("DIFFERENTIAL TESTING: Parallel QE vs Z3 Binary QE")
    print("="*80)

    results = []

    # Test cases
    test_cases = [
        ("Simple Linear", z3.And(z3.Real("x") > 0, z3.Real("x") < 10),
         [z3.Real("x")]),
        ("Multiple Vars",
         z3.And(z3.Real("x") + z3.Real("y") > 5,
                z3.Real("x") - z3.Real("y") < 3),
         [z3.Real("x")]),
        ("Disjunctive",
         z3.Or(z3.Real("x") > z3.Real("y") + 1,
               z3.Real("x") < z3.Real("y") - 1),
         [z3.Real("x")]),
        ("Unsatisfiable",
         z3.And(z3.Real("x") > 5, z3.Real("x") < 3),
         [z3.Real("x")]),
        ("Tautology",
         z3.Or(z3.Real("x") > 0, z3.Real("x") <= 0),
         [z3.Real("x")]),
        ("Boolean",
         z3.And(z3.Implies(z3.Bool("p"), z3.Bool("q")), z3.Bool("p")),
         [z3.Bool("p")]),
    ]

    for name, formula, qvars in test_cases:
        results.append(run_single_test(name, formula, qvars))

    # Summary
    total = len(results)
    matches = sum(1 for r in results if r['match'])

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total}")
    print(f"Matches: {matches} ({matches/total*100:.1f}%)")
    print(f"Differences: {total-matches}")

    if total > matches:
        print("\nDifferences found in:")
        for r in results:
            if not r['match']:
                print(f"  - {r['name']}")

    return results


if __name__ == "__main__":
    differential_test()
