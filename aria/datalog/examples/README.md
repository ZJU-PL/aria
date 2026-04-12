# pyDatalog Examples

These examples exercise the vendored `pyDatalog` runtime in `aria.datalog`,
imported via `from aria.datalog import py_datalog`.

They target the trimmed ARIA vendor rather than a full upstream mirror.

Run them from the repo root with:

```bash
python3 -m aria.datalog.examples.tutorial_example
python3 -m aria.datalog.examples.datalog_example
python3 -m aria.datalog.examples.pythonic_api_example
python3 -m aria.datalog.examples.graph_example
python3 -m aria.datalog.examples.queens_example
python3 -m aria.datalog.examples.python_objects_example
```

Files:

- `tutorial_example.py`: small recursive-family example
- `datalog_example.py`: pure Datalog employee facts and aggregates
- `pythonic_api_example.py`: builder-style API with explicit relations/rules/queries and structured results
- `graph_example.py`: graph reachability and shortest-path style queries
- `queens_example.py`: 8-queens encoding
- `python_objects_example.py`: querying Python objects through Datalog rules
