# Contributing to ARIA

Thank you for your interest in contributing to ARIA! This document outlines the process for contributing.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Recommended: uv (https://docs.astral.sh/uv/)

### Setup Development Environment

```bash
git clone https://github.com/ZJU-PL/aria
cd aria
bash setup_local_env.sh
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest aria/srk/tests/test_*.py

# Run with coverage
pytest --cov=aria
```

## Code Style

### Formatting

```bash
# Format code
black aria/
isort aria/

# Check formatting
black --check aria/
isort --check-only aria/
```

### Type Checking

```bash
mypy aria/
```

### Linting

```bash
flake8 aria/
pylint aria/
```

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Commit

```bash
git add .
git commit -m "feat: add new feature"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/).

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Guidelines

### Python 3.8+

Use type hints:
```python
def my_function(x: int, y: str) -> bool:
    ...
```

### Docstrings

Use Napoleon (Google-style) docstrings:
```python
def my_function(x: int, y: str) -> bool:
    """Short description.

    Args:
        x: Description of x
        y: Description of y

    Returns:
        Description of return value
    """
    ...
```

### Error Handling

- Use specific exceptions
- Don't use bare `except:`
- Log errors appropriately

## Adding New Modules

1. Create module directory in `aria/`
2. Add `__init__.py` with public exports
3. Add to `aria/__init__.py` `__all__` list
4. Add tests in `aria/module/tests/`
5. Update documentation in `docs/source/`

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Include expected vs actual behavior
- Include Python version and OS

## Questions?

Open an issue or contact the maintainers.
