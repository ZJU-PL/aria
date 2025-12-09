# TODO LIST

## Research

There are many interesting research topics. You are welcome to contribute to the following research topics:

- Parallel CDCL(T): See `aria/smt/pcdclt`. Here we have a new algorithm for parallelizing lazy SMT solving
- LLM-driven constraint solving: See `aria/llm`

## Features



### Logic Programming

- https://github.com/pythological/kanren


### Quadratic Programming

- https://github.com/qpsolvers/qpsolvers
- Multi-objective optimization support
- Mixed-integer quadratic programming


### Linear Programming


### Projection and QE

- Fourier-Motzkin elimination
- Simplex method
- Virtual substitution techniques
- Cylindrical algebraic decomposition
- ...

### Advanced Profiling

- https://github.com/viperproject/smt-scope: A tool for visualising, analysing and understanding quantifier instantiations made via E-matching in a run of an SMT solver
- Solver runtime and memory usage analysis
- Constraint complexity metrics
- Visualization of search space exploration
- Performance bottleneck identification

### Parallelization and Distribution

- Distributed solving on clusters
- GPU-accelerated constraint solving
- Portfolio solving with diverse configurations

## Documentation

See `aria/docs`

- Interactive tutorials and examples
- API reference with complete examples
- Performance guidelines and optimization tips

## Applications
You are welcome to contribute to the following applications:

### General

- `Testing`: CIT, symbolic execution, translation validation
- `Static Bug Finding`: value-flow analysis, path-sensitive data-flow analysis
- `Verification`: K-induction, BMC, temlate-based verification, symbolic abstraction
- `Synthesis`: enumerative, deductive, CEGIS, PBE, ...
- `Optimization`: superoptimization, egraph, polyhedral compilation, ...

### Domain Specific

- `Cryptanalysis`, e.g., https://github.com/ranea/CASCADA, https://github.com/kste/cryptosmt, https://github.com/hadipourh/zero
- `ML/LLM`: kernel fusion, verification of neural networks, symbolic reasoning for LLMs
- `Program Repair`: automatic bug fixing with constraint solving
- `Security`: vulnerability detection, exploit generation, formal verification of security protocols, deobfuscation (e.g., for MBA)
- `Planning and Scheduling`: constraint-based AI planning
- `Optimization`: equality saturation, etc.
