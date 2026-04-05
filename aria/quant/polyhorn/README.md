# PolyHorn / PolyQEnt

`aria.quant.polyhorn` is the ARIA area for experiments around polynomial
quantified reasoning.

This line of work is closely related to **PolyQEnt: A Polynomial Quantified
Entailment Solver**:

- Paper: <https://arxiv.org/pdf/2408.03796>
- Upstream project: <https://github.com/ChatterjeeGroup-ISTA/polyqent>

## What PolyQEnt solves

PolyQEnt targets **polynomial quantified entailments (PQE)** of the form

```text
∃t. ∧ᵢ (∀x. Φᵢ(x, t) ⇒ Ψᵢ(x, t))
```

where `t` are existentially quantified template variables and `x` are
universally quantified variables. These problems arise in
**template-based synthesis** for program verification.

## Core idea

The paper's workflow is:

1. parse PQE problems from **SMT-LIB** input,
2. translate them into a canonical form,
3. eliminate universal quantifiers with positivity-theorem-based reductions,
4. solve the resulting existential constraints with SMT backends.

The main positivity results used are:

- **Farkas' lemma**
- **Handelman's theorem**
- **Putinar's theorem**

The implementation discussed in the paper uses backends such as **Z3** and
**MathSAT5**.

## Benchmark context

The paper evaluates this approach on several benchmark families, including:

- termination analysis,
- non-termination analysis,
- almost-sure termination,
- polynomial program synthesis.

Its reported takeaway is that eliminating quantifier alternation with these
positivity-theorem reductions can significantly improve runtime compared with
directly handing the quantified formulas to general-purpose solvers.

## Notes

- `polyhorn` appears to be an older name in this area; the paper and upstream
  project use **PolyQEnt**.
- External solver support matters here; experiments may depend on tools such as
  Z3 or MathSAT5 being available.
