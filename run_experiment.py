import argparse
import os
import subprocess
import yaml
import logging
from datasets import load_dataset, load_from_disk, concatenate_datasets

def setup_logging(log_path):
    """Sets up logging to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def run_command(command):
    """Runs a command and logs its output."""
    logging.info(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        logging.info(line.strip())
    process.wait()
    if process.returncode != 0:
        logging.error(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
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
    dev_data_path = config['dev_data_path']
    test_data_path = config['test_data_path']
    skip_llm_training = config.get('skip_llm_training', False)

    # Create a unique experiment directory
    exp_name = f"experiment_{llm_params['model_name'].replace('/', '_')}_lr{llm_params['learning_rate']}_epochs{llm_params['num_epochs']}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(exp_dir, "pipeline.log")
    setup_logging(log_path)

    llm_training_dir = os.path.join(exp_dir, "llm_training")
    os.makedirs(llm_training_dir, exist_ok=True)

    deberta_training_dir = os.path.join(exp_dir, "deberta_training")
    os.makedirs(deberta_training_dir, exist_ok=True)

    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"Experiment directory created: {exp_dir}")

    # --- Initial Data Conversion ---
    logging.info("\n--- Step 0: Converting data to Hugging Face format ---")
    hf_data_dir = os.path.join(exp_dir, "hf_data")
    os.makedirs(hf_data_dir, exist_ok=True)

    train_hf_path = os.path.join(hf_data_dir, "train")
    dev_hf_path = os.path.join(hf_data_dir, "dev")
    test_hf_path = os.path.join(hf_data_dir, "test")

    if not os.path.exists(train_hf_path):
        train_dataset = load_dataset('json', data_files=train_data_path, split='train')
        train_dataset.save_to_disk(train_hf_path)

    if not os.path.exists(dev_hf_path):
        dev_dataset = load_dataset('json', data_files=dev_data_path, split='train')
        dev_dataset.save_to_disk(dev_hf_path)

    if not os.path.exists(test_hf_path):
        test_dataset = load_dataset('json', data_files=test_data_path, split='train')
        test_dataset.save_to_disk(test_hf_path)
    logging.info("Data converted and saved in Hugging Face format.")

    merged_labeled_train_path = os.path.join(llm_training_dir, "labeled_train_full_hf")
    labeled_dev_path_hf = os.path.join(llm_training_dir, "labeled_dev_hf")

    if not skip_llm_training:
        # --- 1. Divide dataset ---
        logging.info("\n--- Step 1: Dividing dataset ---")
        fold1_path = os.path.join(llm_training_dir, "train_fold1_hf")
        fold2_path = os.path.join(llm_training_dir, "train_fold2_hf")
        run_command(["python", "divide_dataset.py", train_hf_path, fold1_path, fold2_path])
        logging.info("Dataset divided successfully.")

        # --- 2. LLM Training and Inference on Folds ---
        logging.info("\n--- Step 2: LLM Training and Inference on Folds ---")

        def train_and_infer_llm(train_fold_path, infer_fold_path, output_dir_name, llm_params):
            fold_dir = os.path.join(llm_training_dir, output_dir_name)
            os.makedirs(fold_dir, exist_ok=True)

            # Train LLM
            logging.info(f"Training LLM on {train_fold_path}...")
            model_output_dir = os.path.join(fold_dir, "adapter")
            run_command([
                "python", "llm_tuning_secondhalf.py",
                train_fold_path,
                model_output_dir,
                str(llm_params['learning_rate']),
                str(llm_params['num_epochs'])
            ])
            logging.info("LLM training complete.")

            # Merge LoRA
            logging.info("Merging LoRA adapter...")
            merged_model_dir = os.path.join(fold_dir, "merged_model")
            checkpoints = [d for d in os.listdir(model_output_dir) if d.startswith("checkpoint-")]
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            lora_adapter_path = os.path.join(model_output_dir, latest_checkpoint)
            run_command(["python", "save_merged_lora.py", lora_adapter_path, merged_model_dir])
            logging.info("LoRA merging complete.")

            # vLLM Inference
            logging.info(f"Running inference on {infer_fold_path}...")
            predictions_csv_path = os.path.join(fold_dir, "predictions.csv")
            run_command([
                "python", "vllm_inference.py",
                infer_fold_path,
                predictions_csv_path,
                merged_model_dir
            ])
            logging.info("Inference complete.")

            # Postprocess predictions
            logging.info("Postprocessing predictions...")
            labeled_infer_fold_path = os.path.join(fold_dir, "labeled_infer_fold_hf")
            llm_predictions_path = os.path.join(fold_dir, "llm_predictions.jsonl")
            run_command([
                "python", "postprocess_llm.py",
                infer_fold_path,
                predictions_csv_path,
                labeled_infer_fold_path,
                llm_predictions_path,
                "train"
            ])
            logging.info("Postprocessing complete.")
            return labeled_infer_fold_path

        labeled_fold2_path = train_and_infer_llm(fold1_path, fold2_path, "fold1_on_fold2", llm_params)
        labeled_fold1_path = train_and_infer_llm(fold2_path, fold1_path, "fold2_on_fold1", llm_params)

        # --- 3. Merge labeled folds ---
        logging.info("\n--- Step 3: Merging labeled folds ---")
        dataset1 = load_from_disk(labeled_fold1_path)
        dataset2 = load_from_disk(labeled_fold2_path)
        merged_dataset = concatenate_datasets([dataset1, dataset2])
        merged_dataset.save_to_disk(merged_labeled_train_path)
        logging.info("Labeled folds merged successfully.")

        # --- 5. Train final LLM and predict on dev and test ---
        logging.info("\n--- Step 5: Training final LLM and predicting on dev and test data ---")
        final_llm_dir = os.path.join(exp_dir, "final_llm")
        os.makedirs(final_llm_dir, exist_ok=True)

        # Train LLM on full training data
        logging.info("Training final LLM on full training data...")
        final_model_output_dir = os.path.join(final_llm_dir, "adapter")
        run_command([
            "python", "llm_tuning_secondhalf.py",
            train_hf_path,
            final_model_output_dir,
            str(llm_params['learning_rate']),
            str(llm_params['num_epochs'])
        ])
        logging.info("Final LLM training complete.")

        # Merge LoRA
        logging.info("Merging final LoRA adapter...")
        final_merged_model_dir = os.path.join(final_llm_dir, "merged_model")
        checkpoints = [d for d in os.listdir(final_model_output_dir) if d.startswith("checkpoint-")]
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        lora_adapter_path = os.path.join(final_model_output_dir, latest_checkpoint)
        run_command(["python", "save_merged_lora.py", lora_adapter_path, final_merged_model_dir])
        logging.info("Final LoRA merging complete.")

        # --- Process Dev Set ---
        logging.info("\n--- Processing Dev Set with Final LLM ---")
        dev_predictions_csv_path = os.path.join(final_llm_dir, "dev_predictions.csv")
        run_command(["python", "vllm_inference.py", dev_hf_path, dev_predictions_csv_path, final_merged_model_dir])
        final_llm_dev_preds_path = os.path.join(results_dir, f"llm_dev_preds_lr{llm_params['learning_rate']}_epochs{llm_params['num_epochs']}.jsonl")
        run_command(["python", "postprocess_llm.py", dev_hf_path, dev_predictions_csv_path, labeled_dev_path_hf, final_llm_dev_preds_path, "train"])
        logging.info("Dev set processing complete.")

        # --- Process Test Set ---
        logging.info("\n--- Processing Test Set with Final LLM ---")
        test_predictions_csv_path = os.path.join(final_llm_dir, "test_predictions.csv")
        run_command(["python", "vllm_inference.py", test_hf_path, test_predictions_csv_path, final_merged_model_dir])
        labeled_test_path_hf = os.path.join(final_llm_dir, "labeled_test_hf")
        final_llm_test_preds_path = os.path.join(results_dir, f"llm_test_preds_lr{llm_params['learning_rate']}_epochs{llm_params['num_epochs']}.jsonl")
        run_command(["python", "postprocess_llm.py", test_hf_path, test_predictions_csv_path, labeled_test_path_hf, final_llm_test_preds_path, "test"])
        logging.info("Test set processing complete.")

    else:
        logging.info("--- Skipping LLM training as per config ---")
        if not os.path.exists(merged_labeled_train_path) or not os.path.exists(labeled_dev_path_hf):
            raise FileNotFoundError(f"LLM training was skipped, but required labeled files were not found. Please run the full pipeline first or place the files manually.")

    # --- 4. Train DeBERTa on LLM-labeled data ---
    logging.info("\n--- Step 4: Training DeBERTa model ---")
    deberta_exp_name = f"deberta_lr{deberta_params['learning_rate']}_epochs{deberta_params['num_epochs']}"
    deberta_output_dir = os.path.join(deberta_training_dir, deberta_exp_name)
    run_command([
        "python", "transformer_baseline.py",
        "--model_path", deberta_params['model_path'],
        "--train_file", merged_labeled_train_path,
        "--dev_file", labeled_dev_path_hf,
        "--do_train",
        "--do_eval", # Ensure evaluation is done
        "--do_predict",
        "--output_dir", deberta_output_dir,
        "--logging_dir", os.path.join(deberta_output_dir, "logs"),
        "--num_train_epochs", str(deberta_params['num_epochs']),
        "--per_device_train_batch_size", str(deberta_params['train_batch_size']),
        "--per_device_eval_batch_size", str(deberta_params['eval_batch_size']),
        "--learning_rate", str(deberta_params['learning_rate']),
        "--test_files", test_hf_path
    ])
    logging.info("DeBERTa training and prediction complete.")

    # Move deberta predictions to results folder
    deberta_preds_source = os.path.join(deberta_output_dir, "predictions", "test") # The file will be named 'test' based on the basename of the test_hf_path
    deberta_preds_dest = os.path.join(results_dir, f"deberta_preds_lr{deberta_params['learning_rate']}_epochs{deberta_params['num_epochs']}.jsonl")
    os.rename(deberta_preds_source, deberta_preds_dest)
    logging.info(f"DeBERTa predictions moved to {deberta_preds_dest}")

if __name__ == "__main__":
    main()
