# Boolean (SAT) Samplers

This directory contains sampling algorithms for Boolean formulas (SAT).

## Implemented Samplers

### `base.py` - BooleanSampler
Simple enumeration-based sampler using blocking clauses.

**Algorithm:** Generates models by solving and blocking each found solution.

**Pros:**
- Simple and reliable
- Works with any SAT solver

**Cons:**
- Not uniform sampling
- Limited scalability

## Planned/TODO Samplers

### UniGen
Uniform sampling using XOR-based hashing.

**Reference:** Chakraborty et al., "A Scalable Approximate Model Counter", CP 2013

### pyunigen
Python wrapper for the UniGen3 uniform sampler.

**Reference:** https://github.com/meelgroup/pyunigen
