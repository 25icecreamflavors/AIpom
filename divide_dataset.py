import sys
from datasets import load_from_disk

if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py input_hf_dataset_path output_fold1_path output_fold2_path"
        )
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_fold1 = sys.argv[2]
    output_fold2 = sys.argv[3]

    # Loading the train data from disk
    train_dataset = load_from_disk(input_path)
    
    # Shuffling and choosing 2 parts of it
    train_dataset = train_dataset.shuffle(seed=42)
    fold_size = len(train_dataset) // 2
    train1 = train_dataset.select(range(fold_size))
    train2 = train_dataset.select(range(fold_size, len(train_dataset)))
    
    # Saving back to disk
    train1.save_to_disk(output_fold1)
    train2.save_to_disk(output_fold2)