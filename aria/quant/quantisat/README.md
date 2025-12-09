# quanti-sat
New Quantifier Elimination Techniques for SMT Solvers

AAAI 2025: "Quantified Linear and Polynomial Arithmetic Satisfiability via Template-based Skolemization".


## Quick Start
1. Install the required packages. We recommend using a virtual environment with python 3.11. You can install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

2. In order to run all the baselines, including `cvc5`, the commandline tool `cvc5` is required. Find more information [here](https://cvc5.github.io/docs/cvc5-1.0.0/installation/installation.html)

3. Add cnf formulas and negations to the `examples/` directory

4. Run the solver
```bash
python3 main.py --file_pos examples/polysynth_cnf/archimedes.c.smt2 --file_neg examples/polysynth_cnf_neg/archimedes.c.smt2 --solver QUANTISAT --config configs/farkas-z3.json --degree 0 --verbose RESULT_FORMULA_MODEL --timeout 5
```

## Run Experiments
In order to run the experiments, we have used a HPC cluster in order to parallelize. However, you can run the experiments locally in sequence by running the following command:
```bash
./run_all.sh
```
