# Aria Examples

This directory contains examples demonstrating various applications and functionalities of the aria library across different domains.

## Core API Examples (`aria-api/`)

Comprehensive examples showcasing aria's core capabilities using Z3 expressions:
- **AllSMT**: Enumerate all satisfying models of SMT formulas
- **Model Sampling**: Advanced sampling techniques for solution space exploration
- **Unsat Cores**: Compute unsatisfiable cores and minimal unsatisfiable subsets (MUS)
- **Backbone**: Compute backbone literals (literals true in all models)
- **Quantifier Elimination**: Eliminate quantifiers from formulas
- **Abduction**: Find explanations/hypotheses for implications
- **Model Counting**: Count the number of satisfying models
- **Interpolation**: Compute interpolants between formulas
- **MaxSMT**: Solve Maximum Satisfiability problems
- **SyGuS**: Syntax-guided synthesis of functions
- **Unification**: Term unification and pattern matching

## Domain-Specific Applications

### Formal Verification (`validation/`)
- **Model Checking**: Bounded model checking for temporal properties
- **Concurrency Debugging**: Verification of concurrent systems
- **Citation Checking**: Formal verification of academic citations

### Security & Access Control (`access-control/`)
- **ABAC**: Attribute-Based Access Control formal verification
- **RBAC**: Role-Based Access Control modeling
- **Cloud Permissions**: Cloud resource access policy verification

### Compiler Construction (`compiler/`)
- **Graph Coloring**: Constraint solving for register allocation and graph coloring problems

### Cryptanalysis (`crypto/`)
- **CryptoSMT**: Tool for cryptanalysis of symmetric primitives (block ciphers, hash functions, stream ciphers)
- Supports primitives like Simon, Speck, AES variants, Keccak, ChaCha, and many others
- Features differential cryptanalysis, linear cryptanalysis, and key recovery

### Causal Discovery (`cisan/`)
- **CISan**: Runtime verification of causal discovery algorithms using automated conditional independence reasoning
- Implements PC algorithm and variants with SMT-based independence testing
- Research artifact from ICSE 2024 paper

### Games & Puzzles (`games/`)
- **Sudoku**: Constraint solving for Sudoku puzzles

### DSL Development (`pc_dsl/`)
- **Easy Z3 DSL**: Python DSL that simplifies Z3 constraint solving through class-based syntax
- Supports basic types, bit-vectors, strings, arrays, quantifiers, and floating-point numbers
