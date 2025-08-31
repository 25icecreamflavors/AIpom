import pandas as pd
from datasets import load_from_disk, Dataset, concatenate_datasets
import sys

def find_start_index(list_A, common_sublist):
    # Join the words in list_A to form a space-separated string
    joined_str = " ".join(list_A)
    # Join the words in common_sublist to form a space-separated string
    common_str = " ".join(common_sublist)

    # Find the starting index of common_str in joined_str
    start_index = joined_str.find(common_str)

    # If common_str is not found, return -1
    if start_index == -1:
        return -1

    # Count the number of words before the starting index
    words_before_start = joined_str[:start_index].split(" ")
    if words_before_start[-1] == "":
        return len(words_before_start) - 1
    else:
        return len(words_before_start)

def longest_common_sublist(list_A, list_B):
    m, n = len(list_A), len(list_B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list_A[i - 1] == list_B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i - 1
            else:
                dp[i][j] = 0

    common_sublist = list_A[end_index - max_length + 1 : end_index + 1]
    start_word_index = find_start_index(list_A, common_sublist)

    return common_sublist, start_word_index


def post_process(row):
    pred_raw = row.get("prediction_text", "")
    if pd.isna(pred_raw):
        pred_raw = ""
    pred = str(pred_raw).lower()

    if "answer:" not in pred:
        return "none"

    original_text = str(row.get("text", "")).lower().split()   # split() лучше чем split(" ")
    prediction_text = pred.split()

    sublist, _ = longest_common_sublist(original_text, prediction_text)

    if not sublist:
        return "none"
    return sublist

def get_label(row):
    if row["postprocessed"] == "none":
        return -1   
    text_words = str(row["text"]).lower().split()
    label = find_start_index(text_words, row["postprocessed"])
    return label

def add_break(row):
    text = str(row.get("text", ""))
    label = row.get("label_predicted", None)

    try:
        label = int(label)
    except (TypeError, ValueError):
        return text

    text_words = text.split()
    if label == 0:
        return text
    if label < 0 or label >= len(text_words):
        return text

    text_words[label] = "<BREAK>" + text_words[label]
    return " ".join(text_words).rstrip()


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 6:
        print(
            "Usage: python script.py input_hf_dataset_path llm_preds.csv output_hf_dataset_path output_preds_path.jsonl test_mode"
        )
        sys.exit(1)

    # Extract command-line arguments
    input_file_path = sys.argv[1]
    llm_preds_path = sys.argv[2]
    output_train_path = sys.argv[3]
    output_preds_path = sys.argv[4]
    test_mode = sys.argv[5]

    # Load original dataset from disk
    original_dataset = load_from_disk(input_file_path)

    # Load a file with llm predictions
    df = pd.read_csv(llm_preds_path)
    df["text"] = original_dataset["text"]
    
    # Postprocessing step to get the LLM's predicted label
    df["postprocessed"] = df.apply(post_process, axis=1)
    df["label_predicted"] = df.apply(get_label, axis=1)

    # Adding <BREAK> inside the original text for the decoder, based on LLM's prediction
    df["text_deberta"] = df.apply(add_break, axis=1)

    # Save predictions of LLM to check them etc
    sub = df[["id", "label_predicted"]]
    sub = sub.rename(columns={"label_predicted": "label"})
    sub[["id", "label"]].to_json(
        output_preds_path, orient="records", lines=True
    )

    # Create datasets for DeBERTa training/inference
    if test_mode != "test":
        # In train/dev mode, we expect a 'label' column.
        if 'label' not in original_dataset.column_names:
            raise ValueError(f"Input dataset for train mode must contain a 'label' column, but not found in {input_file_path}")
        
        df["original_label"] = original_dataset["label"]

        # Create Augmented Dataset 
        augmented_dataset = Dataset.from_dict({
            "id": df["id"],
            "text": df["text_deberta"],
            "label": df["original_label"] # Use original label
        })
        
        # --- Create Original Dataset (for augmentation) ---
        original_dataset_for_concat = original_dataset.select_columns(["id", "text", "label"])

        # --- Concatenate for Data Augmentation ---
        # The final training data for DeBERTa is the combination of original texts and augmented texts
        final_decoder_dataset = concatenate_datasets([original_dataset_for_concat, augmented_dataset])
        final_decoder_dataset.save_to_disk(output_train_path)

    else:
        # For test mode, we only need the augmented text for inference.
        # No labels are used or expected.
        decoder_dataset_inference = Dataset.from_dict({
            "id": df["id"],
            "text": df["text_deberta"]
        })
        decoder_dataset_inference.save_to_disk(output_train_path)
