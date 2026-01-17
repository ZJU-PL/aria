# SMTO: Satisfiability Modulo Theories and Oracles

The `aria/ml/llm/smto` module implements SMTO that leverages large language models as oracle handlers for SMT solving of programs with unspecified components.

## Key Features

- **Oracle Modes**: Blackbox (input-output examples) and whitebox (analyzes code/docs)
- **Multi-provider**: OpenAI and Anthropic LLM support
- **Caching**: Persistent cache for oracle results
- **Oracle Types**: LLM-based, function-based, and external oracles
- **Type Support**: Integers, reals, booleans, strings, bit-vectors, floating points, arrays
- **Explanations**: Detailed solving process logs for debugging

## Usage Examples

### Blackbox Mode

```python
import z3
from aria.llm import OraxSolver, OracleInfo

solver = OraxSolver(provider="openai", model="gpt-4")
oracle = OracleInfo(
    name="strlen",
    input_types=[z3.StringSort()],
    output_type=z3.IntSort(),
    description="Return the length of a string.",
    examples=[
        {"input": {"arg0": "hello"}, "output": "5"},
        {"input": {"arg0": "world!"}, "output": "6"}
    ]
)
solver.register_oracle(oracle)

s = z3.String('s')
strlen = z3.Function('strlen', z3.StringSort(), z3.IntSort())
solver.add_constraint(strlen(s) == z3.Int('length'))
solver.add_constraint(z3.Int('length') > 5)

model = solver.check()
if model:
    print(f"Solution: s = {model[s]}")
```

### Whitebox Mode

```python
import z3
from aria.llm import OraxSolver, WhiteboxOracleInfo, OracleAnalysisMode

solver = OraxSolver(whitebox_analysis=True)

password_oracle = WhiteboxOracleInfo(
    name="check_password",
    input_types=[z3.StringSort()],
    output_type=z3.BoolSort(),
    description="Return true if password meets complexity requirements.",
    analysis_mode=OracleAnalysisMode.SOURCE_CODE,
    source_code="def check_password(password): return len(password) >= 8 and any(c.isdigit() for c in password)",
    examples=[{"input": {"arg0": "password123"}, "output": "false"}],
    external_knowledge=[
        "Existing tests: check_password('abc12345') == True",
        "Docs: Passwords must include at least one digit and be 8+ chars.",
    ],
)
solver.register_oracle(password_oracle)

password = z3.String('password')
check_pw = z3.Function('check_password', z3.StringSort(), z3.BoolSort())
solver.add_constraint(check_pw(password) == True)

model = solver.check()
if model:
    print(f"Valid password: {model[password]}")
```

### Bit-Vector Operations

```python
import z3
from aria.llm import OraxSolver, OracleInfo

solver = OraxSolver()
bv_oracle = OracleInfo(
    name="bitwise_and",
    input_types=[z3.BitVecSort(8), z3.BitVecSort(8)],
    output_type=z3.BitVecSort(8),
    description="Return bitwise AND of two 8-bit inputs.",
    examples=[
        {"input": {"arg0": "#b11001100", "arg1": "#b10101010"}, "output": "#b10001000"}
    ]
)
solver.register_oracle(bv_oracle)

x, y = z3.BitVecs('x y', 8)
bitwise_and = z3.Function('bitwise_and', z3.BitVecSort(8), z3.BitVecSort(8), z3.BitVecSort(8))
solver.add_constraint(bitwise_and(x, y) != z3.BitVecVal(0, 8))

model = solver.check()
if model:
    print(f"x={model[x]}, y={model[y]}")
```

## Advanced Features

### Custom Function Oracles

```python
from aria.llm import OracleInfo, OracleType

def my_oracle(arg0):
    return len(arg0)

oracle = OracleInfo(
    name="my_oracle",
    input_types=[z3.StringSort()],
    output_type=z3.IntSort(),
    description="Return the length of the input string.",
    oracle_type=OracleType.FUNCTION,
    function=my_oracle
)
```

### Persistent Caching & Debugging

```python
solver = OraxSolver(cache_dir="/path/to/cache")
explanations = solver.get_explanations()
```

## Requirements

- Z3 SMT Solver
- LLM provider: OpenAI (`pip install openai`) or Anthropic (`pip install anthropic`)
