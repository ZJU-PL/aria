# Z3 Tactic Optimization System

A modular implementation of Z3 tactic optimization using genetic algorithms.

## Architecture

- **`core.py`** - Core classes: `Param`, `Tactic`, `TacticSeq`, `GA`, `TacticEvaluator`
- **`utils.py`** - File I/O utilities
- **`main.py`** - CLI interface

## Features

- **Two evaluation modes**: Python API (default) or binary Z3 executable
- **Binary mode**: Replaces `(check-sat)` with `(apply ...)` in SMT-LIB2 files
- **Backward compatible**: Existing code works via `ga_tactics.py` compatibility layer

## Usage

### CLI
```bash
python main.py                                    # Default (Python API)
python main.py --mode binary                      # Binary Z3 mode
python main.py --generations 50 --population 32   # Custom parameters
python main.py --smtlib-file path/to/problem.smt2 # Custom file
```

### Python API
```python
from ga_tactics import TacticSeq, GA, run_tests, EvaluationMode

# Evaluate a tactic sequence
tactic_seq = TacticSeq.random()
fitness = run_tests(tactic_seq, mode=EvaluationMode.BINARY_Z3)

# Run genetic algorithm
ga = GA(population_size=64)
results = ga.run_evolution(generations=128)
print(f"Best: {results['best_sequence'].to_string()}")
```

## Dependencies

- Python 3.6+
- Z3 Python bindings (for Python API mode)
- Z3 binary (for binary mode)

## Testing

```bash
python test_refactoring.py
```
