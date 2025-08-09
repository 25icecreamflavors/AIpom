import json
import sys
from datasets import load_dataset

if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py input_file.jsonl output_fold1.jsonl output_fold2.jsonl"
        )
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_fold1 = sys.argv[2]
    output_fold2 = sys.argv[3]

    # Loading the train data
    train_dataset = load_dataset('json', data_files=input_file, split='train', num_proc=16)
    
    # Shuffling and choosing 2 parts of it
    train_dataset = train_dataset.shuffle(seed=42)
    fold_size = len(train_dataset) // 2
    train1 = train_dataset.select(range(fold_size))
    train2 = train_dataset.select(range(fold_size, len(train_dataset)))
    
    # Saving back to the jsonl format
    train1.to_json(output_fold1)
    train2.to_json(output_fold2)