from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
import sys


def create_answer(example):
    text = example["text"]
    label = example["label"]

    words = text.split(" ")
    answer = " ".join(words[label:])

    if label == len(words) - 1:
        answer = "None"

    return {"answer": answer}


def create_instruct(example):
    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people to check texts for a generated content. User will send you a mixed text, where the first part is human-written and the second part is machine-generated. Terms in text are separated by ' ' symbol. The text starts after the phrase 'Here is the text: '. Your task is to determine the boundary, where the change occurs. As an output write only the machine-generated part of that text. Separate terms by ' '. If the whole text is human-written, output 'None'.",
        },
        {
            "role": "user",
            "content": f"""As an output write only the machine-generated part of the provided text. Output must start with "Answer: ". Separate terms by ' '. If the whole text is human-written, output 'None'. Here is the text: {example["text"]}""",
        },
        {"role": "assistant", "content": f"""Answer: {example["answer"]}"""},
    ]

    return {"prompt_text": chat}


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.jsonl output_dir")
        sys.exit(1)

    # Extract command-line arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    train_dataset = load_dataset(
        "json", data_files=input_file, split="train", num_proc=16
    )
    train_dataset = train_dataset.map(create_answer, num_proc=16)
    train_dataset = train_dataset.map(create_instruct, num_proc=16)

    # Create a model and a tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Open-Orca/Mistral-7B-OpenOrca", use_cache=False
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        "Open-Orca/Mistral-7B-OpenOrca",
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = "<PAD>"

    train_dataset = train_dataset.map(
        lambda x: {
            "prompt_chat": tokenizer.apply_chat_template(
                x["prompt_text"], tokenize=False, add_generation_prompt=False
            )
        }
    )

    # Create a datacollactor
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="epoch",
        learning_rate=2e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        peft_config=peft_config,
        train_dataset=train_dataset,
        dataset_text_field="prompt_chat",
        data_collator=collator,
        max_seq_length=2700,
        dataset_num_proc=16,
        packing=False,
        args=args,
    )

    trainer.train()
