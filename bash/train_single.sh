#!/bin/bash

python -u src/run.py \
    trainer=womd \
    model=scr_womd \
    datamodule=h5_womd \
    datamodule.data_dir=${PWD}/data/h5_womd_hptr \
    hydra.run.dir=${PWD}/logs/hptr_model \
    resume.checkpoint=${PWD}/logs/hptr_model/checkpoints/last.ckpt
