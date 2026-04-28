# QF_DTLIA Sampling Demos

This directory contains runnable examples for the datatype + linear integer
arithmetic sampler.

Run the property-based testing demo from the repository root:

```bash
python -m aria.sampling.dtlia.examples.property_based_testing_demo
```

The demo encodes a small algebraic expression language as a Z3 datatype, uses
QF_DTLIA constraints to describe valid test inputs, samples diverse structured
values, and checks a deliberately buggy evaluator against a reference
interpreter.

