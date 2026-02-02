# Release Guide

This guide covers how to release ARIA to PyPI.

## Prerequisites

```bash
pip install build twine
```

## Release Steps

### 1. Update Version

Update `version` in `pyproject.toml`:
```toml
[project]
version = "0.X.Y"
```

Update `CHANGELOG.md` with the new version and date.

### 2. Build Distribution

```bash
python -m build
```

This creates:
- `dist/aria-0.X.Y.tar.gz` (source distribution)
- `dist/aria-0.X.Y-*.whl` (wheel files)

### 3. Test Installation

```bash
pip install dist/aria-0.X.Y-*.whl --force-reinstall
python -c "import aria; print(aria.__version__)"
```

### 4. Upload to Test PyPI (Recommended)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ aria
```

### 5. Upload to Production PyPI

```bash
twine upload dist/*
```

## Post-Release

1. Create a GitHub release tag
2. Update documentation if needed
3. Announce on relevant channels

## Troubleshooting

### "File already exists"

Remove old builds:
```bash
rm -rf dist/
python -m build
```

### Native extension build failures

Install build dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev z3 libz3-dev

# On macOS with Homebrew
brew install z3
```
