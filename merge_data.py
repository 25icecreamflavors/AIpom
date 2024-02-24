import json
import sys


def merge_files(file1, file2, output_file):
    # Add all items from file1 to the merged file
    merged_data = []
    with open(file1, "r") as f1, open(output_file, "w") as outfile:
        data1 = [json.loads(line) for line in f1]
        merged_data.extend(data1)
        for item in data1:
            outfile.write(json.dumps(item) + "\n")

    # Check items from file2 and add those not already in the merged file
    with open(file2, "r") as f2:
        data2 = [json.loads(line) for line in f2]
        with open(output_file, "a") as outfile:
            for item2 in data2:
                text2 = item2["text"]
                if text2 not in [item["text"] for item in merged_data]:
                    outfile.write(json.dumps(item2) + "\n")


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python script.py file1.jsonl file2.jsonl output.jsonl")
        sys.exit(1)

    # Extract command-line arguments
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]

    # Merge files
    merge_files(file1, file2, output_file)
