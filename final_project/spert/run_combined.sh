#!/bin/bash
# run_combined.sh
# Trains SpERT on combined ADKG+MDKG data, then evaluates on each test set.
#
# Usage:
#   bash run_combined.sh

set -e  # stop on any error
mkdir -p logs

echo "============================================"
echo "  Step 1: Training on combined dataset"
echo "============================================"
python ./spert.py train --config configs/combined_train.conf 2>&1 | tee logs/combined_train_log.txt

# Extract saved path and create symlink
SAVED_PATH=$(grep "Saved in:" logs/combined_train_log.txt | tail -1 | awk '{print $NF}')
echo "Model saved at: $SAVED_PATH"
ln -sfn $(realpath ${SAVED_PATH}/final_model) data/save/combined/best_model
echo "Symlink updated → data/save/combined/best_model"

echo ""
echo "============================================"
echo "  Step 2: Evaluating on ADKG test set"
echo "============================================"
python ./spert.py eval --config configs/combined_eval_adkg.conf 2>&1 | tee logs/combined_eval_adkg_log.txt

echo ""
echo "============================================"
echo "  Step 3: Evaluating on MDKG test set"
echo "============================================"
python ./spert.py eval --config configs/combined_eval_mdkg.conf 2>&1 | tee logs/combined_eval_mdkg_log.txt

echo ""
echo "✅ All done!"
echo "  Logs saved to:"
echo "    logs/combined_train_log.txt"
echo "    logs/combined_eval_adkg_log.txt"
echo "    logs/combined_eval_mdkg_log.txt"
