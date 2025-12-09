# Model Counting Meets Abstract Interpretation (MCAI)

## Implemented features

- Compute the abstractions of a given SMT-LIB2 file or directory of SMT-LIB2 files
- Count the models of the abstractions of a given formula
- Compare the precision of the abstractions 
- Record the results in a csv file, including the false positive rate and the time taken to compute the abstractions

## Usage

Before running MCAI, you need to build the sharpSAT binary by running the following command in the `bin_solvers` directory:
```bash
$ python3 build_sharpsat.py
```

Run the following command in this directory:

```bash
$ python3 bv_mcai.py --help
usage: bv_mcai.py [-h] (-f FILE | -d DIRECTORY | -g) [-l LOG] [-p PROCESSES] [-c CSV]

Model counting and abstract interpretation analysis

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path to SMT-LIB2 file to analyze
  -d DIRECTORY, --directory DIRECTORY
                        Path to directory containing SMT-LIB2 files
  -g, --generate        Generate random formulas for demo
  -l LOG, --log LOG     Path to log file (optional)
  -p PROCESSES, --processes PROCESSES
                        Number of parallel processes for directory processing
  -c CSV, --csv CSV     Path to save the results in a csv file (optional)
```

You can also use the `script.py` to generate random formulas and run the analysis on them.

## TODO features

Use MCAI to understand static analyses

### Impact of overflow / underflow

- Prevent overflow / underflow for all relevant operations in a given SMT formula
  (by adding additional assertions)
- Analyze the presence of overflow / underflow for all relevant operations in a given SMT formula

## Impact of precision-enhancement strategies

- Disjunctive abstraction?
- Specialized bit-level domains?
- Reduced product of (multiple) domains?
- ...?
