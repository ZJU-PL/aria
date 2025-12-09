# LIA* Solver

Implementation from [sls-reachability](https://github.com/mlevatich/sls-reachability)

Based on VMCAI 2020 paper: "Solving LIA* Using Approximations" by Maxwell Levatich, Nikolaj Bjørner, Ruzica Pisakc, and Sharon Shoham.

Solves BAPA (Boolean Algebra with Presburger Arithmetic) benchmarks by:
1. Interpreting them as set or multiset problems
2. Translating to LIA* 
3. Using over/under-approximations with semi-linear set representations
4. Reducing to LIA satisfiability

## Modules

- `lia_star_solver.py` - Main entry point
- `dsl.py` - Translation from BAPA → multiset → LIA*
- `statistics.py` - Algorithm statistics tracking
- `semilinear.py` - Linear and semilinear set operations
- `interpolant.py` - Interpolant and inductive clause management

## Usage

```bash
python3 lia_star_solver.py -h  # Show options

# Examples:
python3 lia_star_solver.py my_mapa_file.smt2 --mapa --unfold=10
python3 lia_star_solver.py my_bapa_file.smt2 --unfold=2
python3 lia_star_solver.py my_mapa_file.smt2 --mapa -v
```