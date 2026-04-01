Knowledge Compilation
=======================


=======
Introduction to Knowledge Compilation
=======

Knowledge compilation is a technique in artificial intelligence 
that aims to preprocess a knowledge base or a set of constraints
into a more efficient and compact representation that can be used for 
faster reasoning. The idea is to transform the
original representation of the knowledge base into a form 
that is easier to manipulate and reason about, , often involving techniques like decision diagrams.
The compiled representation enables more efficient query answering or problem-solving than the original form.


Key Concepts
-----------

1. **Compilation Target Languages**:
   - Different target languages offer different tradeoffs
   - Each supports different types of queries efficiently
   - Expressiveness vs tractability tradeoff

2. **Compilation Process**:
   - Input knowledge base transformation
   - Structure optimization
   - Query preparation

3. **Query Types**:
   - Model counting
   - Consistency checking
   - Clause entailment
   - Model enumeration

==========
Target Languages
==========


OBDD
------

OBDD stands for Ordered Binary Decision Diagram. It's a compact data structure
used in computer science and artificial intelligence to represent Boolean functions.
An OBDD is a directed acyclic graph where each node represents a Boolean variable
with two outgoing edges labeled 0 and 1, indicating the variable being false or true,
respectively. The nodes are ordered according to a fixed variable ordering, ensuring
that equivalent OBDDs are identical.


DNNF
------

Decomposable Negation Normal Form (DNNF) is a representation of propositional
formulas that allows for tractable query answering while maintaining significant
expressive power.

SDD
-----

Sentential Decision Diagrams (SDD) are a relatively new target language that
combines the benefits of OBDDs and DNNFs.

=======
Compilation Techniques
=======

Bottom-up Compilation
------------------

1. **Process**:
   - Variable ordering selection
   - Subformula compilation
   - Composition of results

2. **Optimizations**:
   - Caching
   - Dynamic reordering
   - Structure sharing

Top-down Compilation
-----------------

1. **Approach**:
   - Recursive decomposition
   - Shannon expansion
   - Component analysis

2. **Techniques**:
   - Dynamic decomposition choice
   - Early termination
   - Constraint propagation

=========
Applications
=========

Automated Planning
---------------

1. **Usage**:
   - Plan representation
   - Goal regression
   - Action compilation

2. **Benefits**:
   - Fast plan validation
   - Efficient plan modification
   - Reusable components

Configuration
-----------

1. **Problems**:
   - Product configuration
   - Resource allocation
   - Constraint satisfaction

2. **Advantages**:
   - Interactive configuration
   - Explanation generation
   - Consistency maintenance

Probabilistic Inference
--------------------

1. **Applications**:
   - Bayesian networks
   - Markov networks
   - Probabilistic programs

2. **Operations**:
   - Weighted model counting
   - Marginal computation
   - MAP inference


==========
References
===========

- Darwiche, A., & Marquis, P. (2002). "A Knowledge Compilation Map."
  Journal of Artificial Intelligence Research, 17, 229-264.
- Darwiche, A. (2011). "SDD: A New Canonical Representation of
  Propositional Knowledge Bases." IJCAI Proceedings-International
  Joint Conference on Artificial Intelligence.
- Muise, C., McIlraith, S. A., Beck, J. C., & Hsu, E. I. (2016).
  "DSHARP: Fast d-DNNF Compilation with sharpSAT."
  Canadian Conference on Artificial Intelligence.