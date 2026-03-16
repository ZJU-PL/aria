# FP

## Dumping FP formuals

NOTE: it seems that Z3 creates many bit-vector variables internally. So, the resulting formulas are often in QF_BVFP.

 efmc --file data/FP/pine-benchmarks/ex5_6_chained_2dom.smt2 \
     --efsmt-solver=cegis \
     --cegis-solver-timeout=1 \
     --cegis-dump-dir=/tmp/challenging_queries \
     --cegis-dump-threshold=5