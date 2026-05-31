# GANSAT — GAN-Guided SMT Solver
## Complete Technical Documentation
### University of Manchester | PhD Software Testing | SMT-COMP '26

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Architecture & Design](#3-architecture--design)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
5. [How It Works — Step by Step](#5-how-it-works--step-by-step)
6. [Installation & Setup](#6-installation--setup)
7. [Running the System](#7-running-the-system)
8. [Training the GAN](#8-training-the-gan)
9. [Evaluating Performance](#9-evaluating-performance)
10. [SMT-COMP '26 Submission Guide](#10-smt-comp-26-submission-guide)
11. [Where GANSAT Can Be Used](#11-where-gansat-can-be-used)
12. [Research Contribution](#12-research-contribution)
13. [Limitations & Future Work](#13-limitations--future-work)
14. [Glossary](#14-glossary)

---

## 1. Project Overview

**GANSAT** is a novel SMT (Satisfiability Modulo Theories) solver that uses a
Generative Adversarial Network (GAN) to predict satisfying variable assignments
before calling Z3 for verification.

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Target logic   | QF_LIA (Quantifier-Free Linear Integer Arithmetic) |
| Competition    | SMT-COMP '26                               |
| Backend solver | Z3 4.12+                                   |
| ML framework   | PyTorch 2.1+                               |
| Language       | Python 3.11                                |
| Platform       | Linux (WSL2 / Docker)                      |

### In one sentence
> The GAN guesses the answer in ~1ms; Z3 checks the guess in ~1ms; if wrong,
> Z3 solves it fully as a fallback — so GANSAT is always correct, and often fast.

---

## 2. Background & Motivation

### What is SMT Solving?

SMT (Satisfiability Modulo Theories) solving answers the question:

> Given a set of logical constraints over variables (integers, reals, bit-vectors, etc.),
> does there exist an assignment of values to variables that satisfies all constraints?

**Example:**
```
x + y = 7
x >= 0
y >= 0
x <= 10
```
Answer: **sat**. Model: x=3, y=4 (or many others).

SMT solvers are used in:
- Software verification (does this program have a bug?)
- Test case generation (what inputs trigger this condition?)
- Hardware design verification
- Security analysis (can an attacker reach this state?)

### Why is it Hard?

Classical SMT solving uses the DPLL(T) algorithm — a systematic search through
possible assignments. For large formulas with many variables and tight constraints,
this search can take seconds, minutes, or be undecidable.

### The Key Insight (GAN + SMT)

A PhD researcher in **Software Testing** uses SMT constraints constantly:
- Path constraints from symbolic execution
- Assertion conditions from test generation
- Loop invariants from program analysis

These formulas come from **the same codebase or program family** — they share
structure, variable ranges, and coefficient patterns.

A GAN trained on these formulas **learns the distribution** of satisfying assignments.
Instead of searching, it **predicts** — like a human expert who "knows the answer"
from experience.

### Connection to Your GAN Research

Your existing GAN work generates test cases that satisfy program constraints.
That is **exactly** SMT solving in disguise:

```
Test case generation:          SMT solving:
  program constraints     ==     SMT formula
  valid test input        ==     satisfying assignment
  GAN generator           ==     GANSAT Generator
  test oracle             ==     Z3 verifier
```

---

## 3. Architecture & Design

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: SMT-LIB 2 file                │
└───────────────────────────┬─────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │    PARSER      │  Parse SMT-LIB syntax
                    │  (parser.py)   │  Extract variables & assertions
                    └───────┬────────┘
                            │  ParsedFormula
                    ┌───────▼────────┐
                    │    ENCODER     │  Formula → 8576-dim float vector
                    │  (encoder.py)  │  Variable bounds + constraint matrix
                    └───────┬────────┘
                            │  numpy array [8576]
              ┌─────────────▼──────────────┐
              │        GAN GENERATOR       │  Predicts 16 candidate assignments
              │          (gan.py)          │  ~1ms on CPU
              └─────────────┬──────────────┘
                            │  16 × [64] tensors
              ┌─────────────▼──────────────┐
              │       Z3 VERIFIER          │  Check each candidate (~1ms each)
              │       (solver.py)          │
              └──────┬──────────┬──────────┘
                     │          │
              ┌──────▼──┐  ┌────▼───────────────┐
              │   SAT   │  │   NONE VERIFIED     │
              │  FAST   │  │   Z3 FULL SOLVE     │  Complete fallback
              │  PATH   │  │   (timeout: 20s)    │
              └─────────┘  └─────────────────────┘
```

### GAN Architecture

```
GENERATOR
─────────────────────────────────────────────────────────
Input:  formula_encoding [8576] + noise [128]  =  [8704]
        ↓
        Linear(8704 → 512) + LayerNorm + LeakyReLU
        ↓
        ResBlock × 4  (512 → 512, with skip connections)
        ↓
        Linear(512 → 64) + Tanh
        ↓
Output: assignment_candidates [64]  (values in [-1, 1])


DISCRIMINATOR
─────────────────────────────────────────────────────────
Input:  formula_encoding [8576] + assignment [64]  =  [8640]
        ↓
        Linear(8640 → 512) + LayerNorm + LeakyReLU
        ↓
        ResBlock × 4  (512 → 512)
        ↓
        Linear(512 → 1)
        ↓
Output: logit score  (sigmoid → probability of being satisfying)
```

### Formula Encoding (8576 dimensions)

```
┌─────────────────────────────────────────────────────────┐
│  Variable Bounds Block          64 vars × 2  =  128 dim │
│    [lb_0, ub_0, lb_1, ub_1, ...]                        │
│    Normalized to [-1, 1] by dividing by 10,000          │
├─────────────────────────────────────────────────────────┤
│  Constraint Block      128 constraints × 66  = 8448 dim │
│    Each constraint row:                                  │
│      [coeff_0, coeff_1, ..., coeff_63, rhs, type]       │
│    type: 0=≤  1==  2=≥  3=≠                             │
│    Coefficients normalized by 10,000                    │
└─────────────────────────────────────────────────────────┘
Total: 128 + 8448 = 8576 dimensions
```

### Why ResBlocks?

Residual blocks prevent vanishing gradients in deep networks and allow the
generator to make incremental refinements to assignments — analogous to how
a human solver makes small adjustments to an initial guess.

### Why LayerNorm (not BatchNorm)?

Formula encodings vary wildly in scale depending on coefficient magnitudes.
LayerNorm normalizes per-sample, making training stable without needing
large batch sizes.

---

## 4. File-by-File Breakdown

### `gansat/parser.py`

**Purpose:** Read SMT-LIB 2 format files and extract structured data.

**Key functions:**

| Function | Input | Output | What it does |
|---|---|---|---|
| `parse_file(path)` | file path | `ParsedFormula` | Reads .smt2 file |
| `parse_string(s)` | SMT-LIB string | `ParsedFormula` | Parses inline string |
| `_collect_variables(assertions)` | Z3 expressions | `{name: Z3Var}` | Finds all variables |
| `_extract_logic(s)` | raw string | `"QF_LIA"` | Reads `set-logic` line |

**`ParsedFormula` dataclass:**
```python
assertions: list    # Z3 constraint objects
variables:  dict    # {"x": z3.Int("x"), "y": z3.Int("y")}
var_names:  list    # ["x", "y"]  (sorted, stable order)
logic:      str     # "QF_LIA"
source:     str     # original SMT-LIB text
```

---

### `gansat/encoder.py`

**Purpose:** Convert a `ParsedFormula` into a fixed-size float vector suitable
for PyTorch input.

**Key functions:**

| Function | Description |
|---|---|
| `encode(formula)` | Main encoding → numpy [8576] |
| `decode_assignment(vec, formula)` | GAN output → `{var: int_value}` dict |
| `feature_dim()` | Returns 8576 |
| `_extract_bounds(formula)` | Scans assertions for variable bounds |
| `_extract_linear_constraints(...)` | Extracts Ax ≤ b style rows |
| `_extract_coefficients(expr, ...)` | Recursively parses coefficient of each variable |

**Encoding constants:**
```python
MAX_VARS        = 64      # maximum variables per formula
MAX_CONSTRAINTS = 128     # maximum constraints per formula
BOUND_CLIP      = 1e4     # clip bounds to [-10000, 10000]
COEFF_CLIP      = 1e4     # clip coefficients similarly
```

---

### `gansat/gan.py`

**Purpose:** Define the Generator and Discriminator neural networks.

**Classes:**

| Class | Role |
|---|---|
| `ResBlock(dim)` | Residual block: Linear→LayerNorm→LeakyReLU→Linear→LayerNorm + skip |
| `Generator` | Formula encoding + noise → assignment vector |
| `Discriminator` | Formula encoding + assignment → satisfying probability |

**Key methods:**

```python
# Generate one assignment
G = Generator()
assignment = G(formula_enc)              # shape: [batch, 64]

# Generate multiple candidates for best-of-N selection
candidates = G.sample(formula_enc, n_samples=16)  # shape: [batch, 16, 64]

# Score a (formula, assignment) pair
D = Discriminator()
logit = D(formula_enc, assignment)       # shape: [batch]
prob  = torch.sigmoid(logit)
```

**Hyperparameters:**
```python
hidden  = 512    # hidden layer dimension
depth   = 4      # number of residual blocks
NOISE_DIM = 128  # latent noise dimension for generator
```

---

### `gansat/solver.py`

**Purpose:** The main solver — combines GAN fast path with Z3 fallback.

**Class `GANSATSolver`:**

```python
solver = GANSATSolver(
    model_path  = "models/gansat.pt",  # pre-trained generator weights
    n_candidates = 16,                  # GAN samples to try before fallback
    timeout_ms   = 20_000,             # total time budget (milliseconds)
    device       = "cpu",              # "cuda" if GPU available
)

result, model, elapsed_ms = solver.solve_file("benchmark.smt2")
result, model, elapsed_ms = solver.solve_string(smtlib_string)
```

**Return values:**
```python
result  : "sat" | "unsat" | "unknown"
model   : {"x": 3, "y": 4}  or  None
elapsed : float  (milliseconds)
```

**Solving strategy:**
```
1. If formula has variables:
   a. Encode formula → 8576-dim vector
   b. Generate 16 candidate assignments via GAN
   c. For each candidate:
      - Substitute values into all assertions via Z3
      - Simplify — if all True → return sat immediately
2. If no candidate works → run Z3 full solve with remaining time budget
3. Return Z3's result (always sound and complete)
```

**`_verify_assignment` function:**
Uses `z3.substitute()` to plug values into the formula and `z3.simplify()` to
evaluate — pure symbolic evaluation, no search. This is ~0.1ms per candidate.

---

### `scripts/download_benchmarks.py`

**Purpose:** Get training data — either from official SMT-LIB or generated synthetically.

**Two modes:**

```bash
# Mode 1: Synthetic (fast, no internet)
python scripts/download_benchmarks.py --synthetic --max 2000

# Mode 2: Real SMT-LIB via git sparse clone
python scripts/download_benchmarks.py --logic QF_LIA --max 5000
```

**Synthetic generator:**
Creates random QF_LIA formulas:
- 2–8 integer variables
- 3–16 linear constraints (random coefficients -5 to 5)
- Random right-hand sides (-20 to 20)
- Random operators (≤, ≥, =)

Saves as valid `.smt2` files in `data/benchmarks/synthetic/`.

---

### `scripts/train.py`

**Purpose:** Train the GAN on SAT instances from the benchmark dataset.

**Training procedure:**

```
For each epoch:
  For each batch of (formula, satisfying_assignment) pairs:

    === Train Discriminator (2 steps per batch) ===
    1. Score real (formula, solution) pairs  → high score target
    2. Score fake (formula, GAN_output) pairs → low score target
    3. Backpropagate BCE loss

    === Train Generator (1 step per batch) ===
    4. Generate fake assignments
    5. Score with Discriminator
    6. Maximize Discriminator score (fool it)
    7. Backpropagate
```

**Dataset construction (`SMTDataset`):**
1. Load each `.smt2` file
2. Run Z3 with 5-second timeout to get satisfying assignment
3. Store (formula_encoding, assignment_encoding) pairs
4. Only SAT instances are kept — UNSAT instances have no positive training signal

**Usage:**
```bash
python scripts/train.py \
  --data    data/benchmarks \
  --out     models/gansat.pt \
  --epochs  50 \
  --batch   64 \
  --lr      1e-4 \
  --device  cuda   # or cpu
```

**Outputs:**
- `models/gansat.pt`       — final generator weights
- `models/gansat_best.pt`  — best generator weights (lowest G loss)
- `models/gansat_history.npy` — loss curves for plotting

---

### `scripts/evaluate.py`

**Purpose:** Compare GANSAT performance vs plain Z3 on a held-out benchmark set.

**Metrics reported:**

| Metric | Description |
|---|---|
| Accuracy | % of benchmarks where GANSAT gives correct answer |
| GAN fast-path wins | # solved via GAN alone (no full Z3 search) |
| Avg GANSAT time (ms) | Mean wall-clock time per benchmark |
| Avg Z3-only time (ms) | Mean Z3 baseline time per benchmark |
| Speedup | Ratio: Z3 time / GANSAT time |

**Usage:**
```bash
python scripts/evaluate.py \
  --data       data/benchmarks \
  --model      models/gansat.pt \
  --candidates 16 \
  --timeout    20000 \
  --max        500
```

---

### `main.py`

**Purpose:** SMT-COMP competition entry point. Reads formula, outputs result.

**Interface (SMT-COMP standard):**
```
Input:   SMT-LIB 2 file path as argument, OR formula via stdin
Output:  "sat" / "unsat" / "unknown"
         If sat: model block with variable values
Exit:    0 for sat/unsat, 1 for unknown/error
```

**Usage:**
```bash
# From file
python main.py benchmark.smt2

# From stdin
echo "(set-logic QF_LIA)..." | python main.py --stdin

# With trained model
python main.py --model models/gansat.pt benchmark.smt2
```

---

### `Dockerfile`

**Purpose:** Package GANSAT as a Docker container for SMT-COMP submission.

```dockerfile
FROM python:3.11-slim
COPY gansat/ requirements.txt main.py models/ ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]
```

**Build and test:**
```bash
docker build -t gansat:v1 .
echo "(set-logic QF_LIA)(declare-fun x () Int)(assert (= x 5))(check-sat)" \
  | docker run --rm -i gansat:v1 --stdin
```

---

### `tests/test_pipeline.py`

**Purpose:** Smoke tests — verify the full pipeline from parser to solver.

| Test | What it checks |
|---|---|
| `test_parser` | Variables correctly extracted from SMT-LIB string |
| `test_encoder` | Output shape and dtype correct |
| `test_decode` | Assignment decoding doesn't crash |
| `test_gan_shapes` | Generator and Discriminator produce correct tensor shapes |
| `test_solver_sat` | Solver correctly identifies SAT formula and returns model |
| `test_solver_unsat` | Solver correctly identifies UNSAT formula |

**Run:**
```bash
python tests/test_pipeline.py
# Expected: [ALL TESTS PASSED]
```

---

## 5. How It Works — Step by Step

### Example Input
```smt2
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (>= x 0))
(assert (<= x 10))
(assert (>= y 0))
(assert (<= y 10))
(assert (= (+ x y) 7))
(check-sat)
```

### Step 1: Parse
```
ParsedFormula:
  variables = {"x": Int("x"), "y": Int("y")}
  var_names = ["x", "y"]
  assertions = [x>=0, x<=10, y>=0, y<=10, x+y==7]
  logic = "QF_LIA"
```

### Step 2: Encode
```
Variable bounds block (128 dims):
  x: lb=0.0, ub=10.0  →  [0.0, 0.001, ...]
  y: lb=0.0, ub=10.0  →  [0.0, 0.001, ...]

Constraint block (8448 dims):
  Row 0: x >= 0    →  [1.0, 0.0, ..., 0.0, 0.0]   type=2
  Row 1: x <= 10   →  [1.0, 0.0, ..., 0.001, 0.0] type=0
  Row 2: y >= 0    →  [0.0, 1.0, ..., 0.0, 0.0]   type=2
  Row 3: y <= 10   →  [0.0, 1.0, ..., 0.001, 0.0] type=0
  Row 4: x+y == 7  →  [1.0, 1.0, ..., 0.0007, 1]  type=1
```

### Step 3: GAN generates 16 candidates
```
Candidate 0: x=6, y=1   ← decoded from tanh output
Candidate 1: x=3, y=4
Candidate 2: x=8, y=-1  ← will fail verification
...
```

### Step 4: Z3 verifies Candidate 0
```python
z3.substitute(x+y==7, [(x, 6), (y, 1)])
→ z3.simplify(6+1==7)
→ True

All 5 assertions → True
→ Return SAT, model={x:6, y:1}
```

### Step 5: Output
```
sat
(model
  (define-fun x () Int 6)
  (define-fun y () Int 1)
)
```

---

## 6. Installation & Setup

### Requirements
- Windows 11 with WSL2 (Ubuntu 22.04) **or** Linux **or** Docker
- Python 3.10+
- 4GB RAM minimum (8GB recommended for training)
- GPU optional but speeds up training 10x

### WSL Setup (Windows)

```bash
# Open WSL terminal
wsl

# Navigate to project
cd "/mnt/d/Project/University of Machester"

# Install Python dependencies
pip install z3-solver torch numpy pysmt networkx tqdm scikit-learn matplotlib

# Verify installation
python tests/test_pipeline.py
```

### Virtual Environment (recommended)

```bash
cd "/mnt/d/Project/University of Machester"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/test_pipeline.py
```

---

## 7. Running the System

### Quick test (no training needed)

```bash
# The solver works immediately using Z3 fallback only
python main.py --stdin << 'EOF'
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (= (+ x y) 10))
(assert (>= x 3))
(assert (>= y 3))
(check-sat)
EOF
```

Expected output:
```
sat
(model
  (define-fun x () Int 3)
  (define-fun y () Int 7)
)
```

### Solve a specific file

```bash
python main.py data/benchmarks/synthetic/synthetic_00001.smt2
```

---

## 8. Training the GAN

### Step 1: Generate training data

```bash
# Synthetic (fast, offline — good for first run)
python scripts/download_benchmarks.py --synthetic --max 2000

# Real SMT-LIB benchmarks (requires internet + git)
python scripts/download_benchmarks.py --logic QF_LIA --max 5000
```

### Step 2: Train

```bash
# On CPU (slower, ~1-2 hours for 50 epochs on 2000 samples)
python scripts/train.py --data data/benchmarks --epochs 50

# On GPU (faster, ~10-20 minutes)
python scripts/train.py --data data/benchmarks --epochs 50 --device cuda

# Quick test run (5 epochs)
python scripts/train.py --data data/benchmarks --epochs 5 --batch 32
```

### Step 3: Monitor training

```
[epoch 001/050] loss_D=0.6931  loss_G=0.6931
[epoch 005/050] loss_D=0.5842  loss_G=0.7103
[epoch 010/050] loss_D=0.4921  loss_G=0.7852
...
[checkpoint] Saved generator → models/gansat.pt
```

Healthy training signs:
- `loss_D` decreases from ~0.69 then stabilizes around 0.4–0.6
- `loss_G` increases slightly then stabilizes (generator getting better)
- Neither loss collapses to 0 (mode collapse) or diverges

---

## 9. Evaluating Performance

```bash
python scripts/evaluate.py \
  --data  data/benchmarks \
  --model models/gansat.pt \
  --max   500
```

Expected output (after good training):
```
==================================================
  GANSAT Evaluation Report
==================================================
  Benchmarks evaluated : 500
  Valid (non-timeout)  : 487
  GANSAT accuracy      : 100.0%
  GAN fast-path wins   : 142
  Z3 fallback used     : 345
  Avg GANSAT time (ms) : 18.3
  Avg Z3-only time (ms): 24.7
  Speedup vs Z3        : 1.35x
==================================================
```

Note: Accuracy must remain 100% (correctness guaranteed by Z3 fallback).
Speedup improves as GAN trains on more domain-specific data.

---

## 10. SMT-COMP '26 Submission Guide

### Preliminary Submission (Due: May 1, 2026)

Submit a comment to SMT-COMP describing your approach:

**Template:**
```
Solver name:   GANSAT
Logic target:  QF_LIA
Approach:      GAN-guided assignment prediction with Z3 verification fallback
Novelty:       First application of GAN to warm-start SMT solving in QF_LIA
Institution:   University of Manchester
Contact:       [your email]

Description:
GANSAT uses a Generative Adversarial Network trained on QF_LIA benchmark
distributions to predict satisfying assignments before invoking Z3. The GAN
generates 16 candidate assignments in ~1ms; Z3 verifies each (~0.1ms). If
no candidate is satisfying, Z3 runs fully as a complete fallback, ensuring
soundness and completeness. The approach exploits the structured distribution
of formulas arising from software testing workflows.
```

### Final Submission (typically July–August 2026)

1. Train on full SMT-LIB QF_LIA benchmarks (300k+ formulas)
2. Build Docker image:
   ```bash
   docker build -t gansat:smtcomp26 .
   docker save gansat:smtcomp26 | gzip > gansat_smtcomp26.tar.gz
   ```
3. Upload via SMT-COMP submission portal
4. Verify with SMT-COMP's StarExec/Azure evaluation framework

### Competition tracks to consider

| Track | Description | GANSAT fit |
|---|---|---|
| Single Query Track | One formula per job | Primary target |
| Incremental Track | Sequences of formulas | Future work |
| Model Validation Track | Must produce correct models | Supported |
| Cloud Track | Parallel cloud instances | Future work |

---

## 11. Where GANSAT Can Be Used

### Academic Research

| Domain | Use case |
|---|---|
| Software Testing | Solving path constraints from symbolic execution |
| Program Verification | Checking loop invariants and pre/post-conditions |
| Automated Test Generation | Generating integer inputs satisfying coverage criteria |
| Constraint Programming | Solving integer linear programs with learned priors |

### Software Testing Workflows (Direct PhD Application)

Your GAN-based test case generation can be directly integrated:

```
Program under test
       ↓
Symbolic Execution (e.g., KLEE, SAGE)
       ↓
Path constraints → SMT-LIB 2 formula
       ↓
GANSAT solver
       ↓
Satisfying assignment → concrete test input
       ↓
Execute program with that input
```

GANSAT trained on your program's path constraints will be **faster** than
generic Z3 because it learns the specific structure of your program's
constraint patterns.

### Industry Applications

| Field | Application |
|---|---|
| Embedded Systems | Verifying safety-critical integer arithmetic |
| Compiler Verification | Checking optimization correctness |
| Database Query Optimization | Constraint satisfaction in query planning |
| Network Configuration | Verifying routing constraints |
| Financial Modeling | Integer portfolio optimization constraints |

### As a Research Baseline

Future researchers working on ML-guided SMT solving can use GANSAT as:
- A reference implementation of the GAN-SMT approach
- A baseline to beat with more advanced architectures (Graph Neural Networks, Transformers)
- A framework for experimenting with different ML architectures

---

## 12. Research Contribution

### What is Novel

1. **First GAN application to QF_LIA solving** — prior ML work on SAT uses
   graph neural networks on Boolean formulas. Integer arithmetic has more
   structure (bounds, coefficients) that GANs can exploit.

2. **Warm-start via prediction** — unlike portfolio solvers that pick between
   complete algorithms, GANSAT predicts the answer directly and uses the solver
   only for verification.

3. **Training from solver output** — the training signal is Z3's model output,
   requiring no human labeling. The approach is self-supervised.

4. **Connection to test generation** — explicitly bridges the GAN-based test
   generation literature with formal SMT solving.

### Publishable Claims

- "GANSAT achieves Nx speedup over Z3 on QF_LIA benchmarks drawn from
  software testing workloads, where training and test formulas share structural
  similarity."
- "The GAN fast-path resolves X% of benchmarks without invoking Z3 search."
- "GANSAT maintains 100% correctness due to Z3 verification fallback."

### Suggested publication venues

- **CAV** (Computer-Aided Verification)
- **TACAS** (Tools and Algorithms for Construction and Analysis of Systems)
- **FMCAD** (Formal Methods in Computer-Aided Design)
- **ISSTA** (International Symposium on Software Testing and Analysis) — fits PhD domain
- **ASE** (Automated Software Engineering)

---

## 13. Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Fixed formula size (MAX_VARS=64) | Formulas with >64 variables truncated | Increase MAX_VARS or use Graph NN |
| QF_LIA only | Cannot handle bit-vectors, reals, arrays | Add logic-specific encoders |
| No incremental solving | Each formula solved independently | Add incremental API |
| Simple coefficient encoding | Loses formula topology | Replace with Graph Neural Network |
| Training requires SAT examples | UNSAT formulas provide no training signal | Add contrastive loss |

### Future Work

1. **Graph Neural Network encoder** — represent formula as a graph where
   nodes are variables/operators and edges are constraint relationships.
   GNN naturally handles variable-size formulas.

2. **Transformer-based architecture** — treat formula as a sequence of tokens
   (like code). Pre-trained on SMT-LIB corpus, fine-tuned per logic.

3. **Multi-logic support** — separate encoders and generators per logic
   (QF_BV, QF_NIA, QF_LRA), selected automatically from `set-logic`.

4. **Reinforcement learning** — train generator with reward = 1 if Z3
   verifies assignment in 0 search steps, 0 otherwise. More direct signal.

5. **Integration with symbolic execution** — use GANSAT as the constraint
   solver inside KLEE or angr, with online learning from each execution run.

---

## 14. Glossary

| Term | Definition |
|---|---|
| SMT | Satisfiability Modulo Theories — decides satisfiability of formulas over background theories |
| QF_LIA | Quantifier-Free Linear Integer Arithmetic — no quantifiers, linear constraints, integer variables |
| SAT | Satisfiable — there exists an assignment that makes all constraints true |
| UNSAT | Unsatisfiable — no such assignment exists |
| SMT-LIB 2 | Standard input format for SMT solvers (`.smt2` files) |
| Z3 | Microsoft Research's SMT solver, used as GANSAT's backend |
| GAN | Generative Adversarial Network — Generator vs Discriminator training |
| DPLL(T) | Davis–Putnam–Logemann–Loveland with Theory — classical SMT algorithm |
| ResBlock | Residual block — neural network layer with skip connection |
| LayerNorm | Layer normalization — normalizes activations per sample |
| SMT-COMP | Annual international competition for SMT solvers |
| Fast path | GANSAT solving via GAN prediction alone, without Z3 search |
| Fallback | Z3 full solve — invoked when GAN candidates all fail verification |
| Formula encoding | Fixed-size float vector representing an SMT formula's structure |
| Warm-start | Providing a solver with an initial candidate solution to check first |

---

*Document generated: April 2026*
*Project: GANSAT — University of Manchester PhD Software Testing*
*Competition: SMT-COMP '26, QF_LIA track*
