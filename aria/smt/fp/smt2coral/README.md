# smt2coral

`smt2coral` is a small python module and set of command line tools that take
constraints in the [SMT-LIBv2 format](http://smtlib.cs.uiowa.edu/) and converts
them into the [constraint language](http://pan.cin.ufpe.br/coral/InputLanguage.html) understood by the
[Coral](http://pan.cin.ufpe.br/coral/index.html) constraint solver.

Note that the constraint langauges of Coral and SMT-LIBv2 only partially
overlap in terms of capability. Therefore only a subset of SMT-LIBv2 is supported.

There are several known restrictions due Coral's language constraint being very crude.

* Only constraints in `QF_FP` logic are supported.
* The following operations are not supported: `ite`, `fp.abs`, `fp.fma`,
  `fp.roundToIntegral`, `fp.min`, `fp.max`, and conversion between
  all sorts (e.g. conversion between Float32 and Float64).
* Only Bool, Float32, and Float32 sorts are supported.
* Several translation of SMT-LIBv2 operators are unsound. smt2coral will
  warn when this occurs.

In addition to the above issues, coral itself (well at least version 0.7)
is very buggy and frequently crashes. This will likely limit the use of this
project, however it was quite fun to write so I can't complain too much ;).

## Tools

## `dump.py`

This tool parses SMT-LIBv2 constraints and then dumps the converted constraints
as text. This is useful for debugging/testing. An error will be raised if
the translation cannot be performed.

## `coral.py`

This is a wrapper for Coral that parses SMT-LIBv2 constraints, invokes coral
and responds in a "mostly" SMT-LIBv2 compliant manner.

Note you need to place `coral.jar` in the same directory as `coral.py`. `coral.jar`
is available at http://pan.cin.ufpe.br/coral/Download.html .

Note that the `coral.py` script will ignore command line arguments it doesn't recognise
and will pass them directly to Coral. This is for using Coral's various solver options.

# Dependencies

* Coral >= 0.7 and its dependencies (e.g. Java).
* Z3 4.6.0 and its python bindings.

For testing

* lit (availabel by running `pip install lit`)
* FileCheck

## Testing

Run

```
lit -vs tests/
```
