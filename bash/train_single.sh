#!/bin/bash

python -u src/run.py \
trainer=womd \
model=scr_womd \
datamodule=h5_womd \
datamodule.data_dir=/export/db/HPTR/h5_womd_data/datasets \
hydra.run.dir='./logs/train_hptr_scr'
