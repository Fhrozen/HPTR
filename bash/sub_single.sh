#!/bin/bash


DATASET_DIR="h5_womd_hptr"

python -u src/run.py \
trainer=womd \
model=scr_womd \
datamodule=h5_womd \
resume=sub_womd \
action=validate \
trainer.limit_val_batches=1.0 \
resume.checkpoint='./logs/train_hptr_scr/last_checkpoint.cpkt' \
datamodule.data_dir=/export/db/HPTR/h5_womd_data/datasets \
hydra.run.dir='./logs/train_hptr_scr'

echo finished at: `date`
exit 0;
