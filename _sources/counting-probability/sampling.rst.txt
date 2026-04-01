Model Sampling
=================================


=========
Introduction
=========

Model sampling, also known as solution sampling or witness enumeration, is the process of generating multiple distinct solutions to a logical formula or constraint system. This technique is crucial in various applications including:

- Test case generation
- Statistical analysis
- Probabilistic inference
- Verification and validation

Model sampling is a challenging problem due to the following factors:
- The complexity of the underlying logical formulas or constraints.
- The need for generating a large number of diverse solutions.
- Ensuring the uniformity and independence of the samples.
- ...


Key Concepts
-----------

1. **Sampling Methods**:
   - Uniform sampling
   - Weighted sampling
   - Stratified sampling
   - Markov Chain Monte Carlo (MCMC)

2. **Quality Metrics**:
   - Uniformity
   - Coverage
   - Diversity
   - Independence

3. **Performance Factors**:
   - Sample generation time
   - Memory usage
   - Solution quality
   - Scalability


==========
Model Sampling in Aria
==========



Please refer to ``alrib/sampling``


   .. code-block:: python

      class UniformSampler:
          def __init__(self, formula):
              self.formula = formula

          def sample(self, n):
              # Generate n uniform samples
              pass


======
Related Work
======

- https://github.com/batchenRothenberg/SMTSampler

======
References
======

- Chakraborty, S., Meel, K. S., & Vardi, M. Y. (2013). A scalable approximate model counter. In International Conference on Principles and Practice of Constraint Programming.

-  Ermon, S., Gomes, C. P., & Selman, B. (2012). Uniform solution sampling using a constraint solver as an oracle. In Uncertainty in Artificial Intelligence.

- Kitchen, N., & Kuehlmann, A. (2007). Stimulus generation for constrained random simulation. In IEEE/ACM International Conference on Computer-Aided Design.
