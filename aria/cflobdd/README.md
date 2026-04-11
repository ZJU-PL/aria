# CFL-OBDD

CFL-Reachability using Ordered Binary Decision Diagrams (OBDDs).

This module provides data structures and algorithms for representing and manipulating
context-free language reachability using OBDDs.

## Components

- `cflobvdd.py`: Main CFL-OBDD data structure implementation
- `bvdd.py`: BDD (Binary Decision Diagram) utilities
- `btor2.py`: BTOR2 parser for bit-vector theories
- `z3interface.py`: Z3 integration for solving
- `bitwuzlainterface.py`: Bitwuzla integration for solving
- `relation.py`: finite relation algebra with composition, elimination, and symbolic CFLOBVDD predicates
- `fixpoint.py`: fixed-point and closure helpers for reachability
- `grammar.py`: grammar-driven nonterminal reachability solving
- `pushdown.py`: balanced call/return reachability helpers
- `witness.py`: witness formatting, tree building, and path extraction utilities

## What is CFL-OBDD?

CFL-OBDD combines:
- **CFL (Context-Free Language)**: Formal language with nested structure
- **OBDD**: Ordered Binary Decision Diagram for efficient Boolean function representation

This is useful for analyzing programs with nested relationships (e.g., matching calls/returns,
balanced parentheses, etc.).

## References

Based on techniques from:
- Reps, "Program analysis using OBDDs" 
- et al., "CFL-reachability" algorithms
