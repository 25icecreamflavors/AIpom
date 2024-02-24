#!/bin/bash
exp_name="exp_deberta12_6ep_3e5"
seed_value=42
python transformer_baseline.py \
  --model_path "microsoft/deberta-v3-large" \
  --train_file "data/subtaskC_train.jsonl" \
  --load_best_model_at_end True \
  --dev_file "data/subtaskC_dev.jsonl" \
  --test_files "data/subtaskC_dev.jsonl" \
  --metric_for_best_model "eval_mean_absolute_diff" \
  --greater_is_better False \
  --do_train True \
  --do_predict True \
  --seed $seed_value \
  --output_dir "./runs/$exp_name" \
  --logging_dir "./runs/$exp_name/logs" \
  --num_train_epochs 6 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --auto_find_batch_size True \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --learning_rate 3e-5 \
  --lr_scheduler_type "cosine"

