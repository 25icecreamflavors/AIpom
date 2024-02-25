# <img src='aipom.png' width='75'>  AIpom at SemEval-2024 Task:8

There are several python scripts. We divided them for the usage convenience, since one might want to run separately some parts of the pipeline. For example, training only a LLM instead of training both models. We write commands with example arguments; paths and file names can and should be changed to your own.

## Pipeline:
1. Divide the dataset on 2 folds.
2. Train a LLM on the 1 fold, label (run the inference) the 2 fold. Repeat the procedure with the other one.
3. Train the LLM on the whole train data, label the test (run the inference).
4. Train the decoder on the labeled train. Run the inference on the labeled test.

## Usage instruction:

### 1. LLM training
0. Upload the data (in JSONL format) into the data folder.
1. You need to divide the dataset into 2 folds to separately train 2 LLM models and label the train data without overfitting. Use:
  ```
  python3 divide_dataset.py YourPath_to_train_dataset.jsonl
  ```

2. Then you need to train 2 models to label the train dataset. You can also you the same script, if you want just to train a single LLM model on the whole train data. To train the LLM model, use this command:
  ```
  python3 llm_tuning_secondhalf.py train_file.jsonl output_model_dir
  ```

### 2. LLM inference

1. Since it was a LORA adapter training, to run the inference, you need to merge the adapter with the model before. To avoid using too much memory, you can choose the adapter that you need. To merge it with the model, use the following:
   ```
   python3 save_merged_lora.py lora_adapter_path merged_model_path
   ```

2. To run the the LLM inference use the following command:  
   ```
   python3 vllm_inference.py dataset_path.jsonl output_predictions_path.csv model_checkpoint_path
   ```
3. After the inference one needs to postprocess the predictions to get the labels, since LLM outputs texts. Also if you want to train the decoder models on texts with predictions, you also should use this script.

   ```
   python3 postprocess_llm.py input_file.jsonl llm_preds_path.csv output_train_file.jsonl output_predictions.jsonl test_mode
   ```
   This script will save the predictions of the LLM in the JSONL format (output_predictions.jsonl), so you can use them to check the results. It also saves the file with inserted predictions into the original texts (output_train_file.jsonl). If you run this script on the test_dataset without labels, please pass "test" as the last argument, otherwise pass "train".

### 3. Decoder training and inference

1. To train the decoder or to run the inderence procedure, please use:
   ```
   bash run_3e5.sh
   ```
   Change this file, as you need. Set the paths to the data. You can also find there various hyperparameters.
