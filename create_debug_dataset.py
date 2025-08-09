import argparse
import os

def create_debug_dataset(input_path, output_path, num_lines):
    """Creates a smaller debug dataset from a jsonl file."""
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            if i >= num_lines:
                break
            outfile.write(line)

def main():
    parser = argparse.ArgumentParser(description="Create a debug dataset.")
    parser.add_argument("--train_data_path", type=str, required=True, help="The path to the training data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="The path to the dev data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="The path to the test data.")
    parser.add_argument("--num_lines", type=int, default=3, help="The number of lines to include in the debug dataset.")
    parser.add_argument("--output_dir", type=str, default="data_debug", help="The directory to save the debug dataset.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    create_debug_dataset(
        args.train_data_path,
        os.path.join(args.output_dir, os.path.basename(args.train_data_path)),
        args.num_lines
    )
    create_debug_dataset(
        args.dev_data_path,
        os.path.join(args.output_dir, os.path.basename(args.dev_data_path)),
        args.num_lines
    )
    create_debug_dataset(
        args.test_data_path,
        os.path.join(args.output_dir, os.path.basename(args.test_data_path)),
        args.num_lines
    )

    print(f"Debug dataset created in {args.output_dir}")

if __name__ == "__main__":
    main()
