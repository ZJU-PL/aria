# Utilities

Common utilities used throughout the Aria project.

## Components

### S-Expressions
- `sexpr.py`: S-expression parser
- `sexpr2.py`: Alternative S-expression utilities

### SMT/Solver Utilities
- `smtlib_solver.py`: SMT-LIB solver interface
- `pysmt_solver.py`: PySMT solver wrapper
- `z3_solver_utils.py`: Z3-specific utilities
- `z3_cp_utils.py`: Z3 constraint programming utilities
- `z3_expr_utils.py`: Z3 expression utilities
- `z3_plus_smtlib_solver.py`: Extended SMT-LIB solver
- `z3_ext_candidate.py`: Z3 extension candidate utilities

### Values and Types
- `values.py`: Value manipulation utilities (BV, FP)
- `types.py`: Type definitions
- `logics.py`: Logic definitions

### Parallel Execution
- `parallel/executor.py`: Lightweight parallel execution
- `parallel/master_slave.py`: Master-slave pattern
- `parallel/producer_consumer.py`: Producer-consumer pattern
- `parallel/fork_join.py`: Fork-join pattern
- `parallel/pipeline.py`: Pipeline pattern
- `parallel/actor.py`: Actor model
- `parallel/dataflow.py`: Dataflow graph
- `parallel/stream.py`: Streaming primitives

### PADS Library (Graph Algorithms)
- `pads/Graphs.py`: Graph utilities
- `pads/DFS.py`: Depth-first search
- `pads/BFS.py`: Breadth-first search
- `pads/LexBFS.py`: Lexicographic BFS
- `pads/BipartiteMatching.py`: Hopcroft-Karp
- `pads/UnionFind.py`: Disjoint set
- `pads/PartitionRefinement.py`: Partition refinement
- `pads/StrongConnectivity.py`: SCC algorithms
- `pads/MinimumSpanningTree.py`: Kruskal's algorithm

### Misc
- `misc.py`: Miscellaneous utilities
- `exceptions.py`: Exception classes
