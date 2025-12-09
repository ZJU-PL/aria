# Enhanced SMT Server

This directory contains an enhanced SMT server that provides an SMT-LIB2 interface to Z3 and other advanced features
from the aria library.

## Overview

The SMT server (`smt_server.py`) is a Python program that can be called via IPC. It can take SMT-LIB2 commands (e.g.,
declare-const, assert, check-sat, push/pop, get-model, get-value, etc.) from another program and respond to those
commands.

The enhanced version adds support for advanced aria features:

- AllSMT (enumerating all satisfying models)
- UNSAT core computation
- Backbone literals computation
- Model counting

## Usage

### Starting the Server

You can start the server manually:

```bash
python -m aria.cli.smt_server
```

The server supports several command-line arguments:

```bash
python -m aria.cli.smt_server --help
```

Available options:

- `--input-pipe PATH`: Path to input pipe (default: /tmp/smt_input)
- `--output-pipe PATH`: Path to output pipe (default: /tmp/smt_output)
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Example with custom pipes and debug logging:

```bash
python -m aria.cli.smt_server --input-pipe /tmp/my_input --output-pipe /tmp/my_output --log-level DEBUG
```

This will create two named pipes:

- `/tmp/smt_input`: For sending commands to the server (or custom path if specified)
- `/tmp/smt_output`: For receiving responses from the server (or custom path if specified)

### Sending Commands

You can send commands to the server by writing to the input pipe:

```bash
echo "declare-const x Int" > /tmp/smt_input
```

And read responses from the output pipe:

```bash
cat /tmp/smt_output
```

### Testing

A test script is provided to demonstrate the functionality:

```bash
python -m aria.cli.test_smt_server
```

The test script will automatically start the SMT server if it's not already running and will shut it down when the tests
are complete.

## Supported Commands

### Basic SMT-LIB2 Commands

- `declare-const <name> <sort>`: Declare a constant with the given name and sort (Int, Bool, Real)
- `assert <expr>`: Assert an expression
- `check-sat`: Check satisfiability of the current assertions
- `get-model`: Get a model for the current assertions
- `get-value <var1> <var2> ...`: Get values of specific variables in the current model
- `push`: Push a new scope onto the stack
- `pop`: Pop the top scope from the stack
- `exit`: Exit the server

### Advanced Commands

- `allsmt [:limit=<n>] <var1> <var2> ...`: Enumerate all satisfying models for the given variables
- `unsat-core [:algorithm=<alg>] [:timeout=<n>] [:enumerate-all]`: Compute unsatisfiable cores
- `backbone [:algorithm=<alg>]`: Compute backbone literals
- `count-models [:timeout=<n>] [:approximate]`: Count satisfying models
- `set-option <option> <value>`: Set server options
- `help`: Show available commands

### Configuration Options

- `:allsmt-model-limit <n>`: Set the maximum number of models to enumerate (default: 100)
- `:unsat-core-algorithm <marco|musx|optux>`: Set the UNSAT core algorithm (default: marco)
- `:unsat-core-timeout <n|none>`: Set the timeout for UNSAT core computation (default: none)
- `:model-count-timeout <n>`: Set the timeout for model counting (default: 60 seconds)

## Examples

### Basic SMT-LIB2 Usage

```
declare-const x Int
declare-const y Int
assert (> x 0)
assert (< y 10)
assert (= (+ x y) 5)
check-sat
get-model
```

### AllSMT Example

```
declare-const a Bool
declare-const b Bool
declare-const c Bool
assert (or a b c)
set-option :allsmt-model-limit 10
allsmt a b c
```

### UNSAT Core Example

```
declare-const p Bool
declare-const q Bool
assert p
assert q
assert (not (or p q))
check-sat
unsat-core
unsat-core :algorithm=musx
```

### Backbone Literals Example

```
declare-const a Bool
declare-const b Bool
assert a
assert (or a b)
backbone
```

### Model Counting Example

```
declare-const x Bool
declare-const y Bool
assert (or x y)
count-models
count-models :approximate
```

## Benefits

- Provides a standard SMT-LIB2 interface to Z3 and other aria features
- Enables interoperability with other tools that use SMT-LIB2
- Extends Z3 with advanced capabilities like AllSMT, UNSAT cores, backbone computation, and model counting
- Useful for program analysis, formal verification, and other applications that require SMT solving 