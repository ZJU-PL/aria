# CHC2C Converter

from https://github.com/ftsrg/chc2c

This project is a Python-based tool that converts Constrained Horn Clauses (CHC) in SMT-LIB format into C code for
verification.
The CHC2C tool is designed to handle both linear and non-linear CHC problems, producing equivalent C code which can then
be used in verification processes.

## Features

- Converts CHC files (in SMT-LIB format) into C code.
- Supports both linear and non-linear CHC transformations.
- Automatically retries with a recursive mapping if necessary.
- Command-line interface for specifying input and output files.

## Prerequisites

- Python 3.12
- [z3-solver](https://pypi.org/project/z3-solver/)

For exact versions, see [requirements.txt](requirements.txt).

Run `pip install -r requirements.txt` to install all python dependencies.

## Installation

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/ftsrg/CHC2C.git
cd CHC2C
```

## Usage

To use the tool, run the script with a CHC file as input. The following arguments are supported:

```bash
./chc2c.py <filename> [-o OUTPUT_FILE] [--recursive]
```

- `<filename>`: The path to the CHC file in SMT-LIB format.
- `-o`, `--out`: (Optional) Specifies the name of the output C file. Defaults to `chc.c`.
- `--recursive`: (Optional) Enables recursive conversion for non-linear CHC problems.

If the tool encounters a recursive CHC problem while using the linear conversion, it will automatically retry using a
recursive mapping.

### Example

```bash
./chc2c.py example.chc -o output.c
```

This command converts the `example.chc` file to C code and writes the output to `output.c`.

An example can be seen in the `sample` folder (both for linear and recursive mapping).

## Further Information

For more information on the CHC format, visit [CHC Competition Documentation](https://chc-comp.github.io/format.html).

## Contributors

* [Levente Bajczi](https://github.com/leventebajczi/)
* [Mihály Dobos-Kovács](https://github.com/as3810t/)

## Publications

```bibtex
@inproceedings{chc2c,
    author = {Bajczi, Levente and Moln\'{a}r, Vince},
    title = {{Solving Constrained Horn Clauses as C Programs with CHC2C}},
    year = {2024},
    isbn = {978-3-031-66148-8},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    doi = {10.1007/978-3-031-66149-5_8},
    booktitle = {Model Checking Software: 30th International Symposium, SPIN 2024, Luxembourg City, Luxembourg, April 8–9, 2024, Proceedings},
    pages = {146–163},
    numpages = {18},
    keywords = {CHC, verification, formal methods, software verification},
    location = {Luxembourg City, Luxembourg}
}
```