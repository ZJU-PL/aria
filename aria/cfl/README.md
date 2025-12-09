# Adpated from "Dynamic Transitive Closure-Based Static Analysis through the Lens of Quantum Search"

The artifact is publicly available at [https://github.com/jiawei-95/tosem-QDTCSA-artifact](https://github.com/jiawei-95/tosem-QDTCSA-artifact). You can clone the artifact repository using the command below:

```sh
git clone https://github.com/jiawei-95/tosem-QDTCSA-artifact.git
```

## Dependencies

* Python 3.9 and pypy 3.9
* qiskit (0.32.1)
  
  ```sh
  pip install qiskit==0.32.1
  ```

## Directory

* *.py files are the source codes of the implementation
* **data** files contain the dataset used in the paper
* **cflres**  and **scres** contains expected results.
* **demo** contains a 100KB graph for an example.

## How to run

### using qiskit to simulate Grover search to test the correctness

**dtc.py** and **cfl_dtc.py** use classical simulation to test the correctness of the quantum search subroutine. It will report a missing item if the subroutine loses a target.

```sh
python3 dtc.py $number
python3 cfl_dtc.py $number
```

The input number can be replaced by an integer with a power of 2, but the running time will be slower when the input is larger.

### estimate the number of quantum iterations

**CFGR.py** and **SCR.py** compare the number of classical iterations and the number of quantum iterations of CFL-reachability and SC-reduction, respectively. (pypy is suggested to replace python, and it may take several days to complete the comparison.)

```sh
python3 CFGR.py data/$filename
python3 SCR.py data/$filename
```

A simple example is **demo/100KB.dot**, which can be run using

```sh
python3 CFGR.py demo/100KB.dot
python3 SCR.py demo/100KB.dot
```
