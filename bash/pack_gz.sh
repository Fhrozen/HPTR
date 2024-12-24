#!/usr/bin/env bash

. ./activate_python.sh


for dset in "training" "validation"; do
    python -u src/pack_pkl_womd.py --dataset="${dset}" \
        --out-dir=./data/pkl_womd_hptr \
        --data-dir=./downloads || exit 1
done
