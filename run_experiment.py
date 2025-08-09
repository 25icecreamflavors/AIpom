import argparse
import os
import subprocess
import yaml

def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")

def main():
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    llm_params = config['llm_params']
    deberta_params = config['deberta_params']
    train_data_path = config['train_data_path']
    test_data_path = config['test_data_path']

    # Create a unique experiment directory
    exp_name = f"experiment_{llm_params['model_name'].replace('/', '_')}_lr{llm_params['learning_rate']}_epochs{llm_params['num_epochs']}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    llm_training_dir = os.path.join(exp_dir, "llm_training")
    os.makedirs(llm_training_dir, exist_ok=True)

    deberta_training_dir = os.path.join(exp_dir, "deberta_training")
    os.makedirs(deberta_training_dir, exist_ok=True)

    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")

    # --- 1. Divide dataset ---
    print("\n--- Step 1: Dividing dataset ---")
    fold1_path = os.path.join(llm_training_dir, "train_fold1.jsonl")
    fold2_path = os.path.join(llm_training_dir, "train_fold2.jsonl")
    run_command(["python", "divide_dataset.py", train_data_path, fold1_path, fold2_path])
    print("Dataset divided successfully.")

    # --- 2. LLM Training and Inference on Folds ---
    print("\n--- Step 2: LLM Training and Inference on Folds ---")

    def train_and_infer_llm(train_fold_path, infer_fold_path, output_dir_name, llm_params):
        fold_dir = os.path.join(llm_training_dir, output_dir_name)
        os.makedirs(fold_dir, exist_ok=True)

        # Train LLM
        print(f"Training LLM on {train_fold_path}...")
        model_output_dir = os.path.join(fold_dir, "adapter")
        run_command([
            "python", "llm_tuning_secondhalf.py",
            train_fold_path,
            model_output_dir,
            str(llm_params['learning_rate']),
            str(llm_params['num_epochs'])
        ])
        print("LLM training complete.")

        # Merge LoRA
        print("Merging LoRA adapter...")
        merged_model_dir = os.path.join(fold_dir, "merged_model")
        # Find the latest checkpoint
        checkpoints = [d for d in os.listdir(model_output_dir) if d.startswith("checkpoint-")]
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        lora_adapter_path = os.path.join(model_output_dir, latest_checkpoint)
        run_command(["python", "save_merged_lora.py", lora_adapter_path, merged_model_dir])
        print("LoRA merging complete.")

        # vLLM Inference
        print(f"Running inference on {infer_fold_path}...")
        predictions_csv_path = os.path.join(fold_dir, "predictions.csv")
        run_command([
            "python", "vllm_inference.py",
            infer_fold_path,
            predictions_csv_path,
            merged_model_dir
        ])
        print("Inference complete.")

        # Postprocess predictions
        print("Postprocessing predictions...")
        labeled_infer_fold_path = os.path.join(fold_dir, "labeled_infer_fold.jsonl")
        llm_predictions_path = os.path.join(fold_dir, "llm_predictions.jsonl")
        run_command([
            "python", "postprocess_llm.py",
            infer_fold_path,
            predictions_csv_path,
            labeled_infer_fold_path,
            llm_predictions_path,
            "train"
        ])
        print("Postprocessing complete.")
        return labeled_infer_fold_path

    labeled_fold2_path = train_and_infer_llm(fold1_path, fold2_path, "fold1_on_fold2", llm_params)
    labeled_fold1_path = train_and_infer_llm(fold2_path, fold1_path, "fold2_on_fold1", llm_params)

    # --- 3. Merge labeled folds ---
    print("\n--- Step 3: Merging labeled folds ---")
    merged_labeled_train_path = os.path.join(llm_training_dir, "labeled_train_full.jsonl")
    with open(merged_labeled_train_path, "w") as outfile:
        for fname in [labeled_fold1_path, labeled_fold2_path]:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    print("Labeled folds merged successfully.")

    # --- 4. Train DeBERTa on LLM-labeled data ---
    print("\n--- Step 4: Training DeBERTa model ---")
    deberta_exp_name = f"deberta_lr{deberta_params['learning_rate']}_epochs{deberta_params['num_epochs']}"
    deberta_output_dir = os.path.join(deberta_training_dir, deberta_exp_name)
    run_command([
        "python", "transformer_baseline.py",
        "--model_path", deberta_params['model_path'],
        "--train_file", merged_labeled_train_path,
        "--dev_file", merged_labeled_train_path, # Using merged train as dev for simplicity
        "--do_train",
        "--do_predict",
        "--output_dir", deberta_output_dir,
        "--logging_dir", os.path.join(deberta_output_dir, "logs"),
        "--num_train_epochs", str(deberta_params['num_epochs']),
        "--per_device_train_batch_size", str(deberta_params['train_batch_size']),
        "--per_device_eval_batch_size", str(deberta_params['eval_batch_size']),
        "--learning_rate", str(deberta_params['learning_rate']),
        "--test_files", test_data_path
    ])
    print("DeBERTa training and prediction complete.")

    # --- 5. Train final LLM and predict on test ---
    print("\n--- Step 5: Training final LLM and predicting on test data ---")
    final_llm_dir = os.path.join(exp_dir, "final_llm")
    os.makedirs(final_llm_dir, exist_ok=True)

    # Train LLM on full training data
    print("Training final LLM on full training data...")
    final_model_output_dir = os.path.join(final_llm_dir, "adapter")
    run_command([
        "python", "llm_tuning_secondhalf.py",
        train_data_path,
        final_model_output_dir,
        str(llm_params['learning_rate']),
        str(llm_params['num_epochs'])
    ])
    print("Final LLM training complete.")

    # Merge LoRA
    print("Merging final LoRA adapter...")
    final_merged_model_dir = os.path.join(final_llm_dir, "merged_model")
    checkpoints = [d for d in os.listdir(final_model_output_dir) if d.startswith("checkpoint-")]
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    lora_adapter_path = os.path.join(final_model_output_dir, latest_checkpoint)
    run_command(["python", "save_merged_lora.py", lora_adapter_path, final_merged_model_dir])
    print("Final LoRA merging complete.")

    # vLLM Inference on test data
    print("Running inference on test data...")
    test_predictions_csv_path = os.path.join(final_llm_dir, "test_predictions.csv")
    run_command([
        "python", "vllm_inference.py",
        test_data_path,
        test_predictions_csv_path,
        final_merged_model_dir
    ])
    print("Inference on test data complete.")

    # Postprocess test predictions
    print("Postprocessing test predictions...")
    labeled_test_path = os.path.join(final_llm_dir, "labeled_test.jsonl")
    final_llm_predictions_path = os.path.join(results_dir, f"llm_preds_lr{llm_params['learning_rate']}_epochs{llm_params['num_epochs']}.jsonl")
    run_command([
        "python", "postprocess_llm.py",
        test_data_path,
        test_predictions_csv_path,
        labeled_test_path,
        final_llm_predictions_path,
        "test"
    ])
    print("Postprocessing of test predictions complete.")

    # Move deberta predictions to results folder
    deberta_preds_source = os.path.join(deberta_output_dir, "predictions", os.path.basename(test_data_path))
    deberta_preds_dest = os.path.join(results_dir, f"deberta_preds_lr{deberta_params['learning_rate']}_epochs{deberta_params['num_epochs']}.jsonl")
    os.rename(deberta_preds_source, deberta_preds_dest)
    print(f"DeBERTa predictions moved to {deberta_preds_dest}")


if __name__ == "__main__":
    main()
