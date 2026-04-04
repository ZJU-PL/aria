# pyDatalog Examples

These examples exercise the vendored `pyDatalog` runtime in `aria.datalog`,
imported via `from aria.datalog import pyDatalog`.

They target the trimmed ARIA vendor rather than a full upstream mirror.

Run them from the repo root with:

```bash
python aria/datalog/examples/tutorial_example.py
python aria/datalog/examples/datalog_example.py
python aria/datalog/examples/graph_example.py
python aria/datalog/examples/queens_example.py
python aria/datalog/examples/python_objects_example.py
```

Files:

- `tutorial_example.py`: small recursive-family example
- `datalog_example.py`: pure Datalog employee facts and aggregates
- `graph_example.py`: graph reachability and shortest-path style queries
- `queens_example.py`: 8-queens encoding
- `python_objects_example.py`: querying Python objects through Datalog rules
