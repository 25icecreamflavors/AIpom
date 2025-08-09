import argparse
from transformers import (
    AutoModel,
    AutoTokenizer)

def download_encoder_weights(encoder_name):
    print(f"Downloading pretrained weights for {encoder_name}...")
    model = AutoModel.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    print(f"Pretrained weights for {encoder_name} downloaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download pretrained weights for models."
    )
    parser.add_argument(
        "model_names",
        nargs="+",
        type=str,
        help="Names of models to download weights for.",
    )
    args = parser.parse_args()

    for name in args.model_names:
        download_encoder_weights(name)