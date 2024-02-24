import json
import math
import sys


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def average_labels(labels1, labels2):
    if len(labels1) != len(labels2):
        raise ValueError("Number of labels in both files must be the same.")

    averaged_labels = [
        int((label1 + label2) / 2) for label1, label2 in zip(labels1, labels2)
    ]
    return averaged_labels


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py input_file1.jsonl input_file2.jsonl output_path.jsonl"
        )
        sys.exit(1)

    # Extract command-line arguments
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    output_file = sys.argv[3]

    data1 = load_jsonl(input_file1)
    data2 = load_jsonl(input_file2)

    labels1 = [item["label"] for item in data1]
    labels2 = [item["label"] for item in data2]

    averaged_labels = average_labels(labels1, labels2)

    output_data = [
        {"id": item1["id"], "label": label}
        for item1, label in zip(data1, averaged_labels)
    ]

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")
