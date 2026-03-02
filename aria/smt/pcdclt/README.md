# Parallel CDCL(T) SMT Solver

A parallel implementation of CDCL(T) (Conflict-Driven Clause Learning with Theory) that leverages multi-core processors to accelerate SMT solving.

## The Parallel Solving Idea

Traditional CDCL(T) checks one Boolean model at a time:
```
SAT solver → model₁ → theory check → conflict? → refine → repeat
```

**This implementation parallelizes theory checking** by processing multiple candidate models simultaneously:

```
                  ┌─→ worker₁: check model₁ ─┐
SAT solver → ───→ ├─→ worker₂: check model₂ ─┤ → collect results → refine
                  └─→ workerₙ: check modelₙ ─┘
```

**Key insight**: If the SAT solver can generate many Boolean models cheaply, we can check them in parallel using multiple CPU cores. When any worker finds a theory-consistent model, we immediately return SAT. Otherwise, we learn from all the unsat cores to refine the Boolean search space more aggressively.

### Why This Works

1. **Boolean solving is fast**: Generating multiple SAT models is cheap (microseconds)
2. **Theory checking is slow**: SMT queries can take milliseconds to seconds
3. **Embarrassingly parallel**: Checking different models requires no coordination
4. **Better learning**: Multiple unsat cores per iteration → stronger blocking clauses
    - TODO: those unsat cores are often different...?

## Quick Start

```python
from aria.smt.pcdclt import solve

formula = """
(set-logic QF_LRA)
(declare-fun x () Real)
(assert (and (> x 5) (< x 3)))
"""

result = solve(formula, logic="QF_LRA")  # Uses all CPU cores
```

## Configuration

```python
# config.py
NUM_SAMPLES_PER_ROUND = 10        # Models to check in parallel per iteration
MAX_T_CHECKING_PROCESSES = 0      # Worker processes (0 = all CPU cores)
SIMPLIFY_CLAUSES = True           # Simplify learned blocking clauses
```

## Algorithm Overview

### 1. Preprocess
- Simplify formula using Z3 tactics
- Convert to CNF
- Build Boolean abstraction: theory atoms `(x > 5)` → Boolean vars `p@1`

### 2. Main Loop
```python
while True:
    # Sample N Boolean models from SAT solver
    models = sat_solver.sample_models(NUM_SAMPLES_PER_ROUND)

    if no_models:
        return UNSAT  # Boolean abstraction exhausted

    # Check all models in parallel
    results = parallel_theory_check(models)

    if any_sat(results):
        return SAT  # Found theory-consistent model

    # Learn blocking clauses from all unsat cores
    for core in results:
        sat_solver.add_clause(negate(core))
```

### 3. Parallel Theory Checking
- Main process: Distributes models to worker processes via queue
- Workers: Independent SMT solver instances check models concurrently
- Early exit: First SAT result terminates all workers immediately

## Example

Formula: `(x > 5) ∧ (x < 3)`

| Iteration | Boolean Models | Theory Results | Learned Clause |
|-----------|----------------|----------------|----------------|
| 1 | `{p@1=T, p@2=T}` | UNSAT (core: both) | `¬p@1 ∨ ¬p@2` |
| 2 | Boolean UNSAT | - | Done: **UNSAT** |

With parallelism (NUM_SAMPLES=3):
- Iteration 1: Check 3 models simultaneously → Learn 3 blocking clauses → Converge faster

## When Parallel Helps

✅ **Good for:**
- Complex theories (expensive SMT queries)
- Large formulas with many satisfying assignments
- Multi-core machines with available CPU

❌ **Less helpful for:**
- Simple formulas (preprocessing solves them)
- Highly constrained problems (few Boolean models to check)
- Single-core environments

## Components

- **`solver.py`**: Main parallel CDCL(T) loop
- **`preprocessor.py`**: Formula simplification and Boolean abstraction
- **`theory_solver.py`**: SMT-LIB interface to external solvers (Z3)
- **`config.py`**: Parallelism and behavior settings

## Requirements

- Python 3.7+
- Z3 solver binary (for theory checking)
- PySAT (for Boolean solving)


## 与 parallel by var 的比较
- 它使用了 ICP, 这个技术使得 theory solver 变得容易求解了.
    - 真的吗? 还是仅仅加速了找某一类 UNSAT 的 case. 
    - 应该是吧, 加了很多信息给 theory solver 就是好使呀, 后面的.
- 如果解不出来可以一直尝试降低 theory solver 的压力.
    - 我们的子任务是**不可递归分割的**, 只有两层的切割 && 迭代式?

## TODO
- 核心问题是 theory solver 在一堆 AND_atomic 的问题中, 表现很差
    - 这种情况和直接交给 z3 差不多, 所以一定跑得慢
    - 一个合理的操作是手动操纵布尔结构...?
        - 我擦, 为什么按变量的数值域划分不会遇到这个问题呢
            - 它也没怎么操纵布尔结构, 只是分类讨论的话子问题也是一样的复杂
            - 或者说...给出形如 `a_3 > 7` 这样的 lemma 非常有助于 theory solver 去求解?
            - 还真是 ?
            - 至少解不出来的时候可以无限细分下去.
- 什么是简单的布尔结构, 是否和 AND_atomic 完全一致?
- 这个东西能不能规约成一种特殊的 cube_and_conquer.
    - 在起步的时候是完全一致的呃啊啊啊啊
        - 都是不能分割, 给不出有效信息, 整出几个 lemma 之后再说
    - 来查查大家怎么优化的
