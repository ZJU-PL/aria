

# Machine Learning Components for ARIA

This directory contains machine learning-based tools for automated reasoning and SMT solving optimization.

## Components

### LLM Components: Large Language Model Integration for Automated Reasoning
AI-powered tools leveraging large language models for enhanced SMT solving and theorem proving capabilities.

**Key Features:**
- Natural language processing for SMT formulas
- LLM-assisted abductive reasoning
- Trigger generation for E-matching
- Specification synthesis for closed-box functions
- Specification synthesis for closed-box functions

**Files:**
- `llm/` - Main LLM integration module
  - `smt2nl.py` - Converts SMT-LIB assertions to natural language
  - `abduction/` - LLM-based abduction for hypothesis generation
  - `ematching/` - LLM-trigger generation for E-matching
  - `smto/` - SMT solver with synthesized specifications
  - `smto/` - SMT solver with synthesized specifications

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

### LLM Components

#### SMT to Natural Language
```bash
# Convert SMT-LIB formula to natural language
python -m aria.ml.llm.smt2nl "(assert (and (> x 5) (<= y 10)))"
# Output: "both x is greater than 5 and y is less than or equal to 10"
```

#### LLM-Based Abduction (Natural Language)
```python
from aria.llmtools import LLM, Logger
from aria.ml.llm.abduction import NLAbductor

# Initialize LLM (requires ARIA_LLM_MODEL env var or use default)
llm = LLM(model_name="gpt-4.1-mini", logger=Logger("abduction.log"), temperature=0.2)
abductor = NLAbductor(llm=llm)

# Provide natural language premise and conclusion
text = """Premise: Alice and Bob each have a positive integer number of apples.
Conclusion: Alice has more than 5 apples, and together they have more than 10 apples."""

# Generate abductive hypothesis
result = abductor.abduce(text)
if result.hypothesis:
    print("Hypothesis (SMT):", [t.sexpr() for t in result.hypothesis.smt_terms])
    print("Hypothesis (NL):", list(result.hypothesis.nl_terms))
```



#### E-Matching Trigger Generation
```python
from aria.ml.llm.ematching import LLMTriggerGenerator

# Generate triggers for a given formula
generator = LLMTriggerGenerator()
triggers = generator.generate_triggers("(assert (forall ((x Int)) (> x 0)))")
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

### LLM Dependencies

- `openai` - OpenAI API client (for GPT models)
- `anthropic` - Anthropic API client (for Claude models)
- `aria.llmtools` - ARIA's LLM utilities module

## Configuration

Each component may require specific configuration files:
- Solver configurations in JSON format
- Dataset definitions and paths
- SMAC3 parameter spaces
- Z3 tactic parameter definitions
