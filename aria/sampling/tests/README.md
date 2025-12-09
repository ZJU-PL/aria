# Sampling Module Tests

Comprehensive unit tests for the `aria.sampling` module.

## Test Files

- **test_base.py** - Tests for base classes (Logic, SamplingMethod, SamplingOptions, SamplingResult)
- **test_factory.py** - Tests for SamplerFactory, create_sampler, and sample_models_from_formula
- **test_bool_sampler.py** - Tests for Boolean formula sampler
- **test_bv_samplers.py** - Tests for bit-vector samplers (base, hash-based, QuickSampler)
- **test_lira_sampler.py** - Tests for linear integer/real arithmetic sampler
- **test_mcmc_sampler.py** - Tests for MCMC (Markov Chain Monte Carlo) sampler
- **test_exceptions.py** - Tests for custom exception classes

## Running Tests

```bash
# Run all sampling tests
pytest aria/sampling/tests/

# Run specific test file
pytest aria/sampling/tests/test_base.py

# Run with verbose output
pytest aria/sampling/tests/ -v

# Run with coverage
pytest aria/sampling/tests/ --cov=aria.sampling
```

## Test Coverage

The tests cover:
- **Core functionality**: Sampler initialization, formula parsing, sample generation
- **Multiple logics**: QF_BOOL, QF_BV, QF_LRA, QF_LIA, QF_LIRA
- **Multiple methods**: Enumeration, MCMC, hash-based, QuickSampler
- **Edge cases**: Unsatisfiable formulas, empty samples, blocking clauses
- **Configuration**: Random seeds, timeouts, custom parameters
- **Error handling**: Invalid inputs, missing initialization, unsupported operations

## Notes

- Tests use `pytest.skip()` for samplers that may not be available in all environments
- Random seed tests verify reproducibility where applicable
- Some tests verify that samples actually satisfy the input constraints
