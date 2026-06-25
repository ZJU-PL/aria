# Exists-Forall Synthesis (`efsyn`)

`efdual` contains heuristic CEGIS solvers for formulas:

```text
exists X . domain_x(X) and forall Y . domain_y(Y) -> predicate(X, Y)
```

The solvers are intentionally incomplete. A found witness is certified before
being returned. By default the CEGIS loop is unbounded (`max_iters=None`) and is
expected to be stopped by an external timeout/driver; if a finite `max_iters` is
provided, exhausting it is reported as `budget-exhausted`, not as UNSAT.

## Files

- `efsyn_common.py`: shared dual-memory CEGIS base.
- `efsyn_int.py`: Int/Real specialization.
- `efsyn_bv.py`: bit-vector specialization.
- `efsyn_simple.py`: standalone/simple version of the same algorithm.

## API Contract

Inputs are Z3 constants and formulas:

- `x_vars`: existential variables.
- `y_vars`: universal variables.
- `predicate`: formula over `X` and `Y`.
- `domain_x`: formula over `X`.
- `domain_y`: formula over `Y`.
- `x_templates`: optional formulas over `X`.
- `initial_x`, `initial_y`: optional concrete assignments.
- `max_iters`: optional finite iteration budget; `None` means unbounded.

For sound definite answers, keep ownership clean: `domain_x` and `x_templates`
should not mention `Y`; `domain_y` should not mention `X`.

If `x_templates` is omitted or empty, it means no template restriction. When
templates are provided, admissible candidates must satisfy:

```text
domain_x(X) and (template_1(X) or ... or template_n(X))
```

## Algorithm

The shared solver is a dual-memory CEGIS loop. It keeps two memories:

- `M_X`: concrete witness candidates.
- `M_Y`: concrete attacks plus generalized failure guards.

Each iteration has four phases:

1. **Witness synthesis (`Y -> X`)**

   Select high-scoring attack bundles from `M_Y`. Solve an SMT problem over `X`
   with hard constraints:

   ```text
   domain_x(X) and selected_template(X) and predicate(X, y_i)
   ```

   for each sampled attack `y_i`. The result is a candidate that satisfies the
   selected known attacks.

2. **Attack synthesis (`X -> Y`)**

   Cluster candidates in `M_X` by their pass/fail signature on known attacks.
   Solve an optimization problem over `Y` with hard constraint:

   ```text
   domain_y(Y)
   ```

   and soft constraints:

   ```text
   not predicate(x_i, Y)
   ```

   for candidates `x_i` in the cluster. If a `Y` breaks at least one candidate,
   it is added to `M_Y`.

3. **Cross-play and scoring**

   Evaluate every candidate against every attack representative. Candidate
   coverage is the fraction of attacks it satisfies; attack power is the
   fraction of candidates it breaks. Both memories are ranked by coverage/power
   plus signature novelty, then pruned to their configured memory limits.

4. **Exact certification**

Certification is the exact query:

```text
domain_y(Y) and not predicate(x, Y)
```

If this is UNSAT and `x` is admissible, the solver returns `status="valid"`.
If it is SAT, the model becomes a new attack.

The CEGIS search is heuristic; only certification is trusted for a positive
answer.

## Soundness

Definite answers are intended to be sound:

- `valid`: a concrete admissible witness was certified.
- `unsat-domain-x`: no admissible `X` exists under `domain_x` and templates.
- `unsat-finite-attacks`: a finite set of concrete `Y` attacks rules out every
  admissible `X`.

Non-definite answers are not proofs:

- `budget-exhausted`: no certified witness found within a finite search budget.
- `unknown`: a required solver query returned unknown.

Failure-region guards are only kept when they imply failure for the candidate;
otherwise the solver falls back to the exact counterexample point.

## Variants

- `LinearExistsForallCEGIS`: uses `QF_LIA` or `QF_LRA` when possible. Witness
  synthesis maximizes arithmetic slack for satisfied sampled attacks; attack
  synthesis maximizes violation magnitude. Failure guards prefer linear
  comparison atoms.
- `BVExistsForallCEGIS`: uses `QF_BV`. Novelty and guards are bit-slice aware:
  counterexamples are generalized with equalities over extracts such as
  `Extract(hi, lo, y) == c`.
- `ArcCEGISExistsForallSolver`: simple standalone implementation of the same
  dual-memory loop without the linear/BV-specific objectives.

## Testing

```bash
python -m pytest aria/quant/efdual -q
```
