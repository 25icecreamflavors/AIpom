import pandas as pd
from datasets import load_dataset
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
    original_text = row["text"].lower().split(" ")
    prediction_text = row["prediction_text"].lower().split(" ")

    if (
        row["prediction_text"].find("Answer:") == -1
        and row["prediction_text"].find("answer:") == -1
    ):
        return "none"

    sublist, _ = longest_common_sublist(original_text, prediction_text)

    return sublist


def get_label(row):
    if row["postprocessed"] == "none":
        return 0

    label = find_start_index(
        row["text"].lower().split(" "), row["postprocessed"]
    )

    return label


def add_break(row):
    text = row["text"]
    label = row["label_predicted"]
    text_words = text.split(" ")

    if label == -1:
        return text
    if label >= len(text_words):
        return text

    text_words[label] = "<BREAK>" + text_words[label]
    return " ".join(text_words).rstrip()


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 6:
        print(
            "Usage: python script.py input_file.jsonl llm_preds.csv output_train.jsonl output_path.jsonl test_mode"
        )
        sys.exit(1)

    # Extract command-line arguments
    input_file = sys.argv[1]
    llm_preds_path = sys.argv[2]
    output_train_jsonl = sys.argv[3]
    output_path_jsonl = sys.argv[4]
    test_mode = sys.argv[5]

    # load original texts to merge them
    test_dataset = load_dataset("json", data_files=input_file, split="train")

    # load a file with llm predictions
    df = pd.read_csv(llm_preds_path)
    df["text"] = pd.DataFrame(test_dataset["text"])

    # Postprocessing step
    df["postprocessed"] = df.apply(post_process, axis=1)
    df["label_predicted"] = df.apply(get_label, axis=1)

    # Adding <break> inside the original text for the decoder
    df["text_deberta"] = df.apply(add_break, axis=1)

    if test_mode == "train":
        # Saving the JSON file to train the decoder
        df_json = df[["id", "label", "text_deberta"]]
        df_json = df_json.rename(columns={"text_deberta": "text"})
        df_json[["id", "text", "label"]].to_json(
            output_train_jsonl, orient="records", lines=True
        )
    else:
        # Saving the JSON file for the decoder inference
        df_json = df[["id", "text_deberta"]]
        df_json = df_json.rename(columns={"text_deberta": "text"})
        df_json[
            [
                "id",
                "text",
            ]
        ].to_json(output_train_jsonl, orient="records", lines=True)

    # Save predictions of LLM to check them etc
    sub = df[["id", "label_predicted"]]
    sub = sub.rename(columns={"label_predicted": "label"})
    sub[["id", "label"]].to_json(
        output_path_jsonl, orient="records", lines=True
    )
