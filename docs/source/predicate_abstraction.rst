Predicate Abstraction
====================

Introduction to Predicate Abstraction
-----------------------------------

Predicate abstraction is a technique for constructing finite-state abstractions
from large or infinite-state systems.

The fundamental operation in predicate abstraction can be summarized as follows:
Given a formula φ and a set of predicates P = {p1, ..., pn} in a theory T, generate the most
precise approximation of φ using P.

Depending on the nature of the problem domain, one may either want to generate:

(i) the best underapproximation of φ, i.e., the weakest Boolean combination
    of P that implies φ (denoted by FP(φ)) or

(ii) the best overapproximation of φ, i.e., the strongest Boolean combination of P
     that is implied by φ (denoted by GP(φ)).

Here, the notions of weakness, strength and implication are with respect
to entailment in the given theory T.

Implementation in aria
----------------------

aria implements predicate abstraction in `aria/symabs/predicate_abstraction/predicate_abstraction.py`. 
The implementation focuses on computing the strongest necessary condition of a formula that is 
a Boolean combination of the given predicates.

### Core Functions

- `eval_predicates(m, predicates)`: Evaluates a set of predicates in a model `m`, determining 
  which are true and which are false.
  
- `prime_implicant(ps, e)`: Finds a subset of predicates that together imply a formula, 
  leveraging unsat cores to minimize the subset.
  
- `check_entailment(antecedent, consequent)`: Checks if one formula entails another using Z3.
  
- `predicate_abstraction(fml, predicates)`: The main function that implements predicate abstraction 
  following the CAV'06 paper "SMT Techniques for Fast Predicate Abstraction". This function 
  computes the strongest Boolean combination of predicates that follows from the input formula.

### Algorithm Implementation

The main predicate abstraction algorithm works as follows:

1. Initialize a Z3 solver with the input formula
2. Maintain a set of results, initially empty
3. While the formula is satisfiable:
   a. Get a model from the solver
   b. Evaluate all predicates in the model
   c. Create a conjunction of the evaluated predicates
   d. Add this conjunction to the results
   e. Add the negation of this conjunction to the solver to exclude it in future iterations
4. Return the disjunction of all accumulated conjunctions

```python
def predicate_abstraction(fml: z3.ExprRef, predicates: List[z3.ExprRef]) -> z3.ExprRef:
    """Compute the strongest necessary condition of fml that is the Boolean combination of preds"""
    s = z3.Solver()
    s.add(fml)
    res = []
    while s.check() == z3.sat:
        m = s.model()
        # Compute a prime/minimal implicant
        projs = z3.And(eval_predicates(m, predicates))
        res.append(projs)
        s.add(negate(projs))

    return z3.simplify(z3.Or(res))
```

Example Usage
------------

Consider the following simple C loop:

```c
int i = 0;
while (i < 10) {
    i++;
}
```
Suppose we choose the following predicates as our abstraction:

```c
P1: i < 10
P2: i >= 10
```

To apply predicate abstraction using aria:

```python
import z3
from aria.symabs.predicate_abstraction.predicate_abstraction import predicate_abstraction

# Define variables
i = z3.Int('i')

# Define the formula (represents the loop body)
formula = z3.And(i >= 0, i < 10, i + 1 >= 0)

# Define predicates
P1 = i < 10
P2 = i >= 10
predicates = [P1, P2]

# Compute predicate abstraction
result = predicate_abstraction(formula, predicates)
print(z3.simplify(result))  # This will output: i < 10
```

The result expresses that the strongest necessary condition of the formula in terms of the
given predicates is simply `i < 10`, which makes sense as this is the condition for the loop
to continue.

Advanced Topics
-------------

### Optimization with Prime Implicants

The `prime_implicant` function attempts to find a minimal subset of predicates that imply a 
formula. This can be used to optimize the predicate abstraction process by reducing the size 
of generated Boolean combinations:

```python
# Get a more minimal representation
projs = z3.And(eval_predicates(m, predicates))
projs = prime_implicant(projs, fml)  # Further optimization
```

### Scalability Considerations

For large formulas or many predicates, predicate abstraction can become computationally expensive. 
Some approaches to improve scalability:

1. Incremental SMT solving to reuse solver state
2. Parallel checking of multiple models
3. Predicate clustering to handle related predicates together
4. Using specialized techniques for specific theories (e.g., linear arithmetic)

References
----------

- [GS97] S. Graf and H. Saidi. Construction of abstract state graphs with PVS. In
  CAV'97
- [CKSY04] E. Clarke, D. Kroening, N. Sharygina, and K. Yorav. Predicate abstraction
  of ANSI–C programs using SAT. FMSD'04
- [LBC03] S. K. Lahiri, R. E. Bryant, and B. Cook. A symbolic approach to predicate
  abstraction. In CAV'03
- [LB04] S. K. Lahiri and R. E. Bryant. Constructing Quantified Invariants via
  Predicate Abstraction. In VMCAI'04
- [CAV'06] Cimatti, A., Griggio, A., & Sebastiani, R. (2006). SMT Techniques for 
  Fast Predicate Abstraction. In CAV'06

