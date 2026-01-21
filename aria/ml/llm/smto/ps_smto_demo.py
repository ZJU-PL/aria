"""Demo: PS_SMTO - SMT Solver with Synthesized Specifications.

This demonstrates the new PS_SMTO solver with:
1. Specification synthesis from code/docs/examples
2. Bidirectional SAT/UNSAT search
3. CDCL-style conflict learning
"""

import z3
from aria.ml.llm.smto import PS_SMTOConfig, PS_SMTOSolver, WhiteboxOracleInfo


def demo_simple_oracle():
    """Demo with a simple max function."""
    
    source_code = '''
def max2(x, y):
    """Returns the maximum of two integers."""
    if x >= y:
        return x
    else:
        return y
'''
    
    config = PS_SMTOConfig(
        model="gpt-4",
        enable_spec_synthesis=True,
        mode=PS_SMTOSolver.SolvingMode.BIDIRECTIONAL,
    )
    
    solver = PS_SMTOSolver(config)
    
    oracle = WhiteboxOracleInfo(
        name="max2",
        input_types=[z3.IntSort(), z3.IntSort()],
        output_type=z3.IntSort(),
        description="Returns the maximum of two integers",
        examples=[
            {"input": {"arg0": 5, "arg1": 3}, "output": 5},
            {"input": {"arg0": -2, "arg1": 7}, "output": 7},
            {"input": {"arg0": 0, "arg1": 0}, "output": 0},
        ],
        source_code=source_code,
    )
    solver.register_oracle(oracle)
    
    x = z3.Int('x')
    y = z3.Int('y')
    max2_func = z3.Function('max2', z3.IntSort(), z3.IntSort(), z3.IntSort())
    
    solver.add_constraint(max2_func(x, y) > 10)
    
    result = solver.check()
    
    print(f"Status: {result.status}")
    if result.status == SolvingStatus.SAT and result.model:
        print(f"Model: x = {result.model[x]}, y = {result.model[y]}")


def demo_unsat_proof():
    """Demo showing UNSAT proof construction."""
    
    source_code = '''
def abs(x):
    """Returns absolute value of x."""
    if x >= 0:
        return x
    else:
        return -x
'''
    
    config = PS_SMTOConfig(
        model="gpt-4",
        enable_spec_synthesis=True,
        mode=PS_SMTOSolver.SolvingMode.UNSAT_ONLY,
    )
    
    solver = PS_SMTOSolver(config)
    
    oracle = WhiteboxOracleInfo(
        name="abs",
        input_types=[z3.IntSort()],
        output_type=z3.IntSort(),
        description="Returns absolute value of x",
        examples=[
            {"input": {"arg0": 5}, "output": 5},
            {"input": {"arg0": -3}, "output": 3},
        ],
        source_code=source_code,
    )
    solver.register_oracle(oracle)
    
    x = z3.Int('x')
    abs_func = z3.Function('abs', z3.IntSort(), z3.IntSort())
    
    solver.add_constraint(abs_func(x) < 0)
    solver.add_constraint(x < 0)
    
    result = solver.check()
    
    print(f"Status: {result.status}")
    if result.status == SolvingStatus.UNSAT:
        print("Proved unsatisfiable!")


def demo_docs_only():
    """Demo with only documentation (no source code)."""
    
    config = PS_SMTOConfig(
        model="gpt-4",
        enable_spec_synthesis=True,
        mode=PS_SMTOSolver.SolvingMode.BIDIRECTIONAL,
    )
    
    solver = PS_SMTOSolver(config)
    
    oracle = WhiteboxOracleInfo(
        name="clamp",
        input_types=[z3.IntSort(), z3.IntSort(), z3.IntSort()],
        output_type=z3.IntSort(),
        description="Clamps x to be between low and high inclusive",
        examples=[
            {"input": {"arg0": 5, "arg1": 0, "arg2": 10}, "output": 5},
            {"input": {"arg0": -5, "arg1": 0, "arg2": 10}, "output": 0},
            {"input": {"arg0": 15, "arg1": 0, "arg2": 10}, "output": 10},
        ],
    )
    solver.register_oracle(oracle)
    
    x = z3.Int('x')
    low = z3.Int('low')
    high = z3.Int('high')
    clamp_func = z3.Function('clamp', z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.IntSort())
    
    solver.add_constraint(clamp_func(x, low, high) == 15)
    solver.add_constraint(low == 0)
    solver.add_constraint(high == 10)
    
    result = solver.check()
    
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")


if __name__ == "__main__":
    from aria.ml.llm.smto import SolvingStatus
    
    print("=" * 50)
    print("PS_SMTO Demo - Specification Synthesis")
    print("=" * 50)
    
    print("\n1. SAT finding with source code:")
    demo_simple_oracle()
    
    print("\n2. UNSAT proof with source code:")
    demo_unsat_proof()
    
    print("\n3. Docs-only mode (no source):")
    demo_docs_only()
