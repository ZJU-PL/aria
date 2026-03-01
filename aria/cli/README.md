# ARIA CLI Tools

Command-line interface tools for automated reasoning tasks.

## Overview

This package provides CLI tools for various automated reasoning tasks:

| Tool | Description |
|------|-------------|
| `fmldoc` | Format conversion, validation, and analysis for logic constraints |
| `mc` | Model counting for Boolean, QF_BV, and arithmetic theories |
| `pyomt` | Optimization modulo theories (OMT) solving |
| `efsmt` | Exists-Forall SMT solving |
| `smt_server` | Enhanced SMT server with advanced features |

## Quick Start

```bash
# Format conversion
python -m aria.cli.fmldoc translate -i input.cnf -o output.smt2

# Model counting
python -m aria.cli.mc formula.smt2

# Optimization
python -m aria.cli.pyomt problem.smt2

# Exists-Forall solving
python -m aria.cli.efsmt problem.smt2

# SMT server
python -m aria.cli.smt_server
```

---

## fmldoc - Format Converter

Convert between logic constraint formats and validate files.

### Commands

```bash
# Translate DIMACS to SMT-LIB2
python -m aria.cli.fmldoc translate -i input.cnf -o output.smt2

# Validate a file
python -m aria.cli.fmldoc validate -i input.smt2 -f smtlib2

# Analyze file properties
python -m aria.cli.fmldoc analyze -i input.cnf

# List supported formats
python -m aria.cli.fmldoc formats

# Batch processing
python -m aria.cli.fmldoc batch -i input_dir/ -o output_dir/
```

### Supported Formats

| Format | Extension | Validate | Analyze | Translate From | Translate To |
|--------|-----------|----------|---------|----------------|--------------|
| DIMACS | .cnf | ✓ | ✓ | ✓ | ✓ |
| SMT-LIB2 | .smt2 | ✓ | ✓ | - | ✓ |

---

## mc - Model Counter

Count satisfying models for formulas.

### Usage

```bash
# Auto-detect theory
python -m aria.cli.mc formula.smt2

# Specify theory
python -m aria.cli.mc formula.cnf --theory bool
python -m aria.cli.mc formula.smt2 --theory bv
python -m aria.cli.mc formula.smt2 --theory arith

# Set timeout
python -m aria.cli.mc formula.smt2 --timeout 300

# Debug output
python -m aria.cli.mc formula.smt2 --log-level DEBUG
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--theory` | bool, bv, arith, auto | auto | Theory to use |
| `--method` | solver, enumeration, auto | auto | Counting method |
| `--timeout` | integer | None | Timeout in seconds |
| `--log-level` | DEBUG, INFO, WARNING, ERROR | INFO | Logging level |

---

## pyomt - Optimization Solver

Solve optimization modulo theories problems.

### Usage

```bash
# Default engine (qsmt)
python -m aria.cli.pyomt problem.smt2

# Specific engine
python -m aria.cli.pyomt problem.smt2 --engine qsmt
python -m aria.cli.pyomt problem.smt2 --engine iter
python -m aria.cli.pyomt problem.smt2 --engine maxsat
python -m aria.cli.pyomt problem.smt2 --engine z3py

# Specify theory
python -m aria.cli.pyomt problem.smt2 --theory bv
python -m aria.cli.pyomt problem.smt2 --theory arith
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--type` | omt, maxsmt | omt | Problem type |
| `--theory` | bv, arith, auto | auto | Theory type |
| `--engine` | qsmt, maxsat, iter, z3py | qsmt | Optimization engine |
| `--solver` | string | auto | Backend solver |
| `--log-level` | DEBUG, INFO, WARNING, ERROR | INFO | Logging level |

**Note:** MaxSMT support is not yet fully implemented.

---

## efsmt - Exists-Forall Solver

Solve Exists-Forall SMT problems.

### Usage

```bash
# Auto-detect theory and engine
python -m aria.cli.efsmt problem.smt2

# Specify parser
python -m aria.cli.efsmt problem.smt2 --parser z3
python -m aria.cli.efsmt problem.smt2 --parser sexpr

# Specify theory
python -m aria.cli.efsmt problem.smt2 --theory bool
python -m aria.cli.efsmt problem.smt2 --theory bv
python -m aria.cli.efsmt problem.smt2 --theory lira

# Use specific engine
python -m aria.cli.efsmt problem.smt2 --engine z3
python -m aria.cli.efsmt problem.smt2 --engine cegar
python -m aria.cli.efsmt problem.smt2 --engine efbv-par

# Set limits
python -m aria.cli.efsmt problem.smt2 --timeout 60 --max-loops 1000
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--parser` | z3, sexpr | z3 | Parsing backend |
| `--theory` | auto, bool, bv, lira | auto | Theory selection |
| `--engine` | auto, z3, cegar, efbv-par, efbv-seq, eflira-par, eflira-seq | auto | Solver engine |
| `--timeout` | integer | None | Timeout in seconds |
| `--max-loops` | integer | None | Max CEGAR iterations |
| `--log-level` | DEBUG, INFO, WARNING, ERROR | INFO | Logging level |

### Input Format

EFSMT problems use SMT-LIB2 syntax with:
- `declare-fun` for existentially quantified variables
- `assert` with `forall` for universal quantification

Example:
```smt2
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (forall ((y Int)) (=> (>= y 0) (>= x y))))
(check-sat)
```

---

## smt_server - Enhanced SMT Server

Run an SMT server with advanced features via IPC.

### Usage

```bash
# Start with defaults
python -m aria.cli.smt_server

# Custom pipes
python -m aria.cli.smt_server --input-pipe /tmp/my_input --output-pipe /tmp/my_output

# Debug mode
python -m aria.cli.smt_server --log-level DEBUG
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-pipe` | /tmp/smt_input | Input pipe path |
| `--output-pipe` | /tmp/smt_output | Output pipe path |
| `--log-level` | INFO | Logging level |

### Basic Commands

| Command | Description |
|---------|-------------|
| `declare-const <name> <sort>` | Declare constant (Int, Bool, Real) |
| `assert <expr>` | Assert expression |
| `check-sat` | Check satisfiability |
| `get-model` | Get satisfying model |
| `get-value <vars...>` | Get variable values |
| `push` / `pop` | Scope management |
| `exit` | Exit server |

### Advanced Commands

| Command | Description |
|---------|-------------|
| `allsmt [:limit=<n>] <vars...>` | Enumerate all models |
| `unsat-core [:algorithm=<alg>]` | Compute unsat cores |
| `backbone [:algorithm=<alg>]` | Compute backbone literals |
| `count-models [:timeout=<n>]` | Count models |
| `set-option <opt> <val>` | Configure server |
| `help` | Show help |

### Example Session

```bash
# Terminal 1: Start server
python -m aria.cli.smt_server

# Terminal 2: Send commands
echo "declare-const x Bool" > /tmp/smt_input
echo "declare-const y Bool" > /tmp/smt_input
echo "assert (or x y)" > /tmp/smt_input
echo "check-sat" > /tmp/smt_input
cat /tmp/smt_output  # sat
```

---

## Error Handling

All CLI tools use consistent error handling:

- **Exit codes:**
  - `0`: Success
  - `1`: Error (message to stderr)

- **Debug mode:** Use `--log-level DEBUG` for stack traces

- **Common errors:**
  - File not found
  - Invalid format
  - Unsupported feature
  - Solver timeout

## Testing

Run CLI tests:

```bash
# All CLI tests
pytest aria/tests/test_cli_*.py

# Specific tool
pytest aria/tests/test_cli_fmldoc.py -v
pytest aria/tests/test_cli_mc.py -v
```

## Dependencies

**Required:**
- Python 3.8+
- z3-solver

**Optional:**
- pysmt (for some OMT engines)
- sharpSAT (Boolean model counting)
- LattE (arithmetic model counting)
- External SMT solvers (cvc5, yices, etc.)

## Configuration

Some tools require external solver configurations:

1. Create `config.json` in project root
2. Set solver paths
3. Or set `ARIA_CONFIG` environment variable

See `config_example.json` for template.

## License

See main project LICENSE. 