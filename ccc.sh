#!/usr/bin/env bash
set -euxo pipefail

# 1. Create a new sqlite3 db file with datetime in the filename
dt=$(date +"%Y%m%d_%H%M%S")
DBFILE="classification_results_${dt}.db"
touch "$DBFILE"

# 2. Run python -m train and python -m ccc 10 times, passing db file to ccc.py
for i in $(seq 1 10); do
    rm -f *.keras 
    python -m train
    python -m ccc --db "$DBFILE"
done
