# ProofMate (LLM4SMT)

Using LLMs to aid in solving SMT files with induction.

## Usage

### Preprocessing
- Preprocess the raw benchmark with `preprocess.py`
    - Processed files are placed under the `preprocessed/` directory
- Put the prepared benchmark in the run directory

### Running
- After preprocessing, use `Mate.py` to run the tests
- Example workflow with run directory `test-ben/test-vmcai15dt`:

```bash
mkdir test-ben/test-vmcai15dt
cp -r preprocessed/vmcai15-dt test-vmcai15dt

# run one case
python3 Mate.py test-vmcai15dt/clam/nosg/goal1

# run directory
bash run_all_benchmarks.sh

# results
python3 result-llm.py
```

### Local environment configuration
Environment setup:
```bash
    ## Install langchain
    pip install langchain_openai python-dotenv 
    ## Install colored logging
    pip install colorlog
```
Create a local `.env` file:
```config
OPENAI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=sk-xxx
# If not set, GPT-4o is used by default
MODEL_TYPE=deepseek
```

## File description
- **Mate.py** — Main program
- **result-llm.py** — Analyzes log files in the specified directory and generates an Excel report
- **preprocessed.py** — SMT2 preprocessing: parsing, refactoring, template generation, and batch processing

## Publication

**Can LLM Aid in Solving Constraints with Inductive Definitions?**

Solving constraints involving inductive (aka recursive) definitions is challenging. State-of-the-art SMT/CHC solvers and first-order logic provers provide only limited support for solving such constraints, especially when they involve, e.g., abstract data types. In this work, we leverage structured prompts to elicit Large Language Models (LLMs) to generate auxiliary lemmas that are necessary for reasoning about these inductive definitions. We further propose a neuro-symbolic approach, which synergistically integrates LLMs with constraint solvers: the LLM iteratively generates conjectures, while the solver checks their validity and usefulness for proving the goal. We evaluate our approach on a diverse benchmark suite comprising constraints originating from algebraic data types and recurrence relations. The experimental results show that our approach can improve the state-of-the-art SMT and CHC solvers, solving considerably more (around 25%) proof tasks involving inductive definitions, demonstrating its efficacy.

**Full version paper:** [https://doi.org/10.48550/arXiv.2603.03668](https://doi.org/10.48550/arXiv.2603.03668)
