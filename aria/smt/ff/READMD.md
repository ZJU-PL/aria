# SMT Solving for Finite Field Arithmetic

This module implements SMT solving capabilities for finite field arithmetic, supporting various theories and solving
techniques.

## Overview

The finite field SMT solver supports:

- Quantifier-free finite field formulas (QF_FF or QF_FFA)

## Features

### Core Capabilities

- Native support for finite field operations (add, mul, div)
- Translation of finite field formulas to bit-vector (QF_BV) or integer arithmetic (QF_NIA)

### Advanced Features

- Gröbner basis computation for algebraic reasoning (via CVC5)
- MCSat-based solving techniques (via Yices2)

## Encoding Principles

The solver translates finite field arithmetic to bit-vector (QF_BV) or integer (QF_NIA) theories for SMT solving. Three encoding strategies are supported:

### Pure Integer Encoding

The most direct encoding represents field elements as Z3 integers with constraints `0 ≤ x < p` for each variable. Operations are performed directly in the integer domain using Z3's native integer arithmetic, with modulo reduction applied after each operation. This approach targets the QF_NIA theory and provides the simplest translation, avoiding bit-width calculations and conversions entirely. However, non-linear integer arithmetic can be challenging for SMT solvers.

### Faithful Bit-Vector Encoding

Field elements are represented as k-bit bit-vectors where k = ⌈log₂ p⌉. To prevent overflow during multiplication, all intermediate computations are performed in a wider 2·k-bit representation. After each operation, results are reduced modulo p using unsigned remainder (`URem`) and extracted back to k bits. This ensures arithmetic correctness while maintaining bit-vector semantics throughout the translation.

### Integer/Bit-Vector Bridge Encoding

An alternative encoding uses Z3's `BV2Int` and `Int2BV` primitives to bridge between bit-vector and integer domains. Field elements are represented as k-bit bit-vectors, but operations are converted to integers, computed in the integer domain, reduced modulo p, and converted back to bit-vectors. This approach leverages Z3's native integer modulo operations, providing a cleaner algebraic representation at the cost of additional conversions.
