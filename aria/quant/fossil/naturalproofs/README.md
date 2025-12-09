# Natural Proofs package

## Requirements

- [Python 3.5 or above](https://www.python.org/downloads/)
- [Z3Py](https://pypi.org/project/z3-solver/)

## Installation (Linux)
 
- Clone the lemma synthesis repo. Alternatively, use GitHub's support for svn to [download only the Natural Proofs package](https://stackoverflow.com/questions/9609835/git-export-from-github-remote-repository/18324428#18324428). 
- Install Z3Py from [PyPI](https://pypi.org/project/z3-solver/) with `pip3 install z3-solver`.
  - You can also build z3 from source with python bindings: [https://github.com/Z3Prover/z3](https://github.com/Z3Prover/z3). Use the `--python` flag while building.
  - Make sure to set the PYTHONPATH and PATH variables correctly as instructed by the z3 installation process.
- Add the path to the naturalproofs toplevel folder to `PYTHONPATH`: execute `export PYTHONPATH ="/path/to/naturalproofs":$PYTHONPATH` or add it to `~/.bashrc` and then do `source ~/.bashrc`.

## Usage

- There are some example files in the `tests` subfolder. Use these to understand how to call the natural proofs solver and configure the options.
- Some automatically generated documentation is available in the `docs` subfolder for further clarification about the API for declaring constants/functions as well as options for the natural proofs solver.
