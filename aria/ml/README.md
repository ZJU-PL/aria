

# Machine Learning Components for ARIA

This directory contains machine learning-based tools for automated reasoning and SMT solving optimization.

## Components

### SMTGazer: Machine Learning-Based SMT Solver Portfolio System
An effective algorithm scheduling method for SMT solving that uses machine learning to select optimal combinations of SMT solvers for different problem categories and instances.

**Key Features:**
- Feature extraction and normalization
- Unsupervised clustering using X-means algorithm
- SMAC3-based portfolio optimization
- Parallel solver execution and evaluation

**Files:**
- `smtgazer/` - Main SMTgazer implementation
- `batchportfolio.py` - Batch processing across SMT categories
- `SMTportfolio.py` - Core ML portfolio system
- `portfolio_smac3.py` - SMAC3 integration for optimization

### MachFea: Machine Learning Feature Extraction
Feature extraction system for SMT problems using the Sibyl feature extractor from MachSMT.

**Key Features:**
- Statistical feature extraction from SMT formulas
- Parallel processing of large problem sets
- Integration with SMTgazer clustering

**Files:**
- `machfea/` - Feature extraction implementation
- `get_feature.py` - Individual problem feature extraction
- `mach_run_inference.py` - Batch feature extraction runner

### TacticGA: Genetic Algorithm for Z3 Tactic Optimization
Genetic algorithm that evolves optimal sequences of Z3 tactics for efficient SMT problem solving.

**Key Features:**
- Population-based evolutionary search
- Configurable tactic sequences
- Fitness evaluation based on solving performance

**Files:**
- `tactic_ga/` - Genetic algorithm implementation
- `ga_tactics.py` - Main genetic algorithm for tactic optimization

## Usage Examples

### SMTGazer Portfolio Training
```bash
# Extract features for training data
cd machfea
python mach_run_inference.py 0

# Train portfolios for specific SMT categories
cd ../smtgazer
python batchportfolio.py  # Trains on Equality+LinearArith, QF_Bitvec, etc.
```

### Feature Extraction
```bash
# Extract features for a single SMT problem
python get_feature.py problem.smt2 --dataset MyDataset

# Batch feature extraction for multiple problems
python mach_run_inference.py <seed>
```

### Tactic Optimization
```python
from tactic_ga.ga_tactics import TacticSeq, GA

# Create and evaluate a tactic sequence
tactic_seq = TacticSeq.random()
print(tactic_seq.to_string())

# Run genetic algorithm (128 generations)
# ga = GA()
# ga.evaluate(); ga.dump(); ga.repopulate()  # Repeat for 128 generations
```

## Publications

- **SMTGazer**: "SMTGazer: Machine Learning-Based SMT Solver Portfolio Selection" (ASE 2025)

## Dependencies

- `pysmt` - Python SMT library
- `z3` - Z3 SMT solver with Python bindings
- `smac3` - Sequential Model-based Algorithm Configuration
- `scikit-learn` - Machine learning utilities
- `numpy` - Numerical computing
- `machsmt` - SMT feature extraction library

## Configuration

Each component may require specific configuration files:
- Solver configurations in JSON format
- Dataset definitions and paths
- SMAC3 parameter spaces
- Z3 tactic parameter definitions
