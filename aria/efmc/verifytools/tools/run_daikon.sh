#! /usr/bin/env bash

if [[ $# -ne 3 ]] ; then
  echo "Usage $0 <path-to-lvlset> <path-to-split-lvlset> <basename-for-csvs>"
  exit -1
fi

echo "Lvlset: $1 to $2"
./run_daikon.py --lvlset $1 --timeout 60 --csv-table --check-solved | tee $3.daikon.nosplit.suppress.csv
./run_daikon.py --lvlset $1 --timeout 60 --csv-table --check-solved --no-suppression | tee $3.daikon.nosplit.nosuppress.csv
./run_daikon.py --lvlset $2 --timeout 60 --use-splitter-predicates --csv-table --check-solved | tee $3.daikon.split.suppress.csv
./run_daikon.py --lvlset $2 --timeout 60 --use-splitter-predicates --csv-table --check-solved --no-suppression | tee $3.daikon.split.nosuppress.csv
