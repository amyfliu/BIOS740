#!/bin/bash
DATASET=$1  # pass "adkg" or "mdkg" as argument

# 1. Train
python ./spert.py train --config configs/${DATASET}_train.conf 2>&1 | tee logs/${DATASET}_train_log.txt

# 2. Auto-extract timestamp and update symlink
SAVED_PATH=$(grep "Saved in:" logs/${DATASET}_train_log.txt | tail -1 | awk '{print $NF}')
ln -sfn $(realpath ${SAVED_PATH}/final_model) data/save/${DATASET}/best_model
echo "Symlink updated → ${SAVED_PATH}/final_model"

# 3. Evaluate
python ./spert.py eval --config configs/${DATASET}_eval.conf 2>&1 | tee logs/${DATASET}_eval_log.txt
