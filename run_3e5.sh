#!/bin/bash
python3 run_experiment.py \
  --train_data_path "data/subtaskC_train.jsonl" \
  --test_data_path "data/subtaskC_dev.jsonl" \
  --learning_rate 2e-5 \
  --num_epochs 5
