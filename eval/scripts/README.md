# Scripts Directory - Optimized Structure

This directory contains format conversion utilities for SMT-LIB and SyGuS files. The scripts have been optimized to reduce redundancy and improve maintainability.

## Main Files

### Core Utilities
- **`conversion_utils.py`** - Shared utility classes and functions for all converters
- **`format_converter.py`** - Unified converter supporting multiple format conversions

### Individual Converters
- **`syguslia2sygusbv.py`** - Convert SyGuS LIA to SyGuS BV (merged signed/unsigned versions)
- **`chclia2sygusbv.py`** - Convert CHC LIA to SyGuS BV
- **`chclia2chcbv.py`** - Convert CHC LIA to CHC BV
- **`sygusbv2chcbv.py`** - Convert SyGuS BV to CHC BV
- **`chcbv2sygusbv.py`** - Convert CHC BV to SyGuS BV
- **`syguslnia2chc.py`** - Convert SyGuS LNIA to CHC LIA

### Other Scripts
- **`chc2c.py`** - Convert CHC to C code
- **`btor2chc.py`** - Convert BTOR2 to CHC
- **`sygsu1to2.sh`** - Shell script to convert SyGuS v1 to v2

## Usage

### Using the Unified Converter
```bash
./format_converter.py <conversion_type> <input_path> <output_dir> [options]
```

Supported conversion types:
- `sygus_lia_to_sygus_bv`
- `chc_lia_to_sygus_bv`
- `chc_lia_to_chc_bv`
- `sygus_bv_to_chc_bv`
- `chc_bv_to_sygus_bv`
- `sygus_lia_to_chc_lia`

Options:
- `--width <int>`: Bitvector width (default: 32)
- `--signedness <signed|unsigned>`: Bitvector signedness (default: signed)

### Using Individual Converters
Each individual converter script supports the same argument pattern:
```bash
python <converter_script.py> <input_path> <output_dir> [--width <int>] [--signedness <signed|unsigned>]
```

## Key Improvements

1. **Reduced Redundancy**: Merged duplicate scripts (e.g., signed/unsigned variants)
2. **Shared Utilities**: Common functions extracted to `conversion_utils.py`
3. **Unified Interface**: Single script handles multiple conversion types
4. **Better Error Handling**: Improved error reporting and file processing
5. **Cleaner Code**: Removed verbose comments and dead code
6. **Modern Python**: Uses f-strings, pathlib, and argparse consistently

## Migration Guide

If you were using the old scripts:
- `syguslia2sygusbv_signed.py` → Use `syguslia2sygusbv.py --signedness signed`
- `syguslia2sygusbv_unsigned.py` → Use `syguslia2sygusbv.py --signedness unsigned`
- Multiple hardcoded conversions → Use `format_converter.py` with appropriate conversion type

## Dependencies

The scripts require:
- Python 3.6+
- z3-solver
- efmc package (from parent directory) 