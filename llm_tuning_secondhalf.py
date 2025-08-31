from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
import sys
import re


def create_answer(example):
    text = example["text"]
    label = example["label"]

    words = text.split(" ")
    
    # Handle edge cases better
    if label >= len(words):
        answer = "None"
    elif label == len(words) - 1:
        answer = "None"
    else:
        answer = " ".join(words[label:])

    return {"answer": answer}


def create_instruct(example):
    # Use DeepSeek's actual chat format
    system_prompt = "You are a helpful assistant that helps people to check texts for a generated content. User will send you a mixed text, where the first part is human-written and the second part is machine-generated. Terms in text are separated by ' ' symbol. The text starts after the phrase 'Here is the text: '. Your task is to determine the boundary, where the change occurs. As an output write only the machine-generated part of that text. Separate terms by ' '. If the whole text is human-written, output 'None'."
    
    user_prompt = f"""As an output write only the machine-generated part of the provided text. Output must start with "Answer: ". Separate terms by ' '. If the whole text is human-written, output 'None'. Here is the text: {example["text"]}"""
    
    assistant_response = f"""Answer: {example["answer"]}"""
    
    # Format for DeepSeek - using the actual format from the logs
    formatted_chat = f"You are a helpful assistant that helps people to check texts for a generated content. User will send you a mixed text, where the first part is human-written and the second part is machine-generated. Terms in text are separated by ' ' symbol. The text starts after the phrase 'Here is the text: '. Your task is to determine the boundary, where the change occurs. As an output write only the machine-generated part of that text. Separate terms by ' '. If the whole text is human-written, output 'None'.\n\nUser: {user_prompt}\n\nAssistant: {assistant_response}"
    
    return {"formatted_chat": formatted_chat}


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python script.py input_file.jsonl output_dir learning_rate num_train_epochs")
        sys.exit(1)

    # Extract command-line arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    learning_rate = float(sys.argv[3])
    num_train_epochs = int(sys.argv[4])
    model_name = "/home/dviazhev/deepseek-llm-7b-chat"
    
    train_dataset = load_from_disk(input_file)
    
    #Use more training examples
    #train_dataset = train_dataset.select(range(5))
    
    train_dataset = train_dataset.map(create_answer, num_proc=16)
    train_dataset = train_dataset.map(create_instruct, num_proc=16)

    # Remove unused columns to avoid conflicts
    columns_to_remove = ['id', 'text', 'label', 'answer']
    existing_columns = train_dataset.column_names
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
    if columns_to_remove:
        train_dataset = train_dataset.remove_columns(columns_to_remove)
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Dataset columns: {train_dataset.column_names}")
    print(f"Sample formatted chat:\n{train_dataset[0]['formatted_chat'][:500]}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        use_cache=False,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # FIX 3: Set correct padding side

    # Use the correct response template for DeepSeek
    # Based on the logs, we need to find the actual assistant response pattern
    sample_text = train_dataset[0]['formatted_chat']
    print(f"Looking for response pattern in sample...")
    
    # Try to find the assistant response pattern
    if "Assistant: Answer:" in sample_text:
        response_template = "Assistant: Answer:"
        print(f"Found response template: '{response_template}'")
    elif "\nAssistant:" in sample_text:
        response_template = "\nAssistant:"
        print(f"Found alternative response template: '{response_template}'")
    else:
        # Fallback - just use the answer part
        response_template = "Answer:"
        print(f"Using fallback response template: '{response_template}'")
    
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"Tokenized template: {response_token_ids}")

    collator = DataCollatorForCompletionOnlyLM(
        response_token_ids,
        tokenizer=tokenizer
    )
    
    # FIX 5: Adjusted training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=4,  
        gradient_checkpointing=True,
        logging_steps=5000,  # Log every step for debugging
        save_strategy="epoch",
        learning_rate=learning_rate,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_drop_last=False,
        remove_unused_columns=True, 
        report_to=None,  # Disable wandb/tensorboard
    )

    peft_config = LoraConfig(
        r=16,  
        lora_alpha=32,
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

    # Test data collator before training
    print("Testing data collator...")
    try:
        # Manually tokenize one sample to test
        sample_input = train_dataset[0]['formatted_chat']
        tokenized = tokenizer(sample_input, return_tensors="pt", truncation=True, max_length=2048)
        
        print(f"Tokenization works. Input shape: {tokenized['input_ids'].shape}")
        
        # Test the data collator with the tokenized sample
        sample_for_collator = [{"input_ids": tokenized['input_ids'][0].tolist()}]
        collated = collator(sample_for_collator)
        
        print(f"Data collator works. Labels shape: {collated['labels'].shape}")
        non_ignore_labels = (collated['labels'] != -100).sum()
        print(f"Non-ignore labels count: {non_ignore_labels}")
        
        if non_ignore_labels == 0:
            print("WARNING: All labels are -100 (ignored). Adjusting response template...")
            # Try a simpler template
            response_template = "Answer:"
            response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
            collator = DataCollatorForCompletionOnlyLM(response_token_ids, tokenizer=tokenizer)
            
    except Exception as e:
        print(f"Data collator test failed: {e}")
        print("Switching to standard data collator...")
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model,
        peft_config=peft_config,
        train_dataset=train_dataset,
        dataset_text_field="formatted_chat",  # Use our simplified field name
        data_collator=collator,
        max_seq_length=1024,  # Reduced for stability
        dataset_num_proc=1,
        packing=False,
        args=args,
    )

    print("Starting training...")
    trainer.train()