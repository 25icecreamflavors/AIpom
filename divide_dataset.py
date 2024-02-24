import json
import sys
from datasets import load_dataset

if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 2:
        print(
            "Usage: python script.py train_dataset.jsonl"
        )
        sys.exit(1)
    
    # Loading the train data
    train_dataset = load_dataset('json', data_files='data/subtaskC_train.jsonl', split='train', num_proc=16)
    
    # Shuffling and choosing 2 parts of it
    train_dataset = train_dataset.shuffle()
    train1 = train_dataset.select(range(1825))
    train2 = train_dataset.select(range(1825, 3649))
    
    # Saving back to the jsonl format
    train1.to_json("train1.jsonl")
    train2.to_json("train2.jsonl")