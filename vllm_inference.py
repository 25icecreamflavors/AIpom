import sys
import pandas as pd
from datasets import load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import json

def load_model_and_tokenizer(model_path, base_model="deepseek-ai/deepseek-llm-7b-chat"):
    """Load model and tokenizer with fallback strategies"""
    print(f"Loading model from {model_path}")
    
    try:
        # Try to load tokenizer from model path first
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Loaded tokenizer from model path")
    except Exception as e:
        print(f" Failed to load tokenizer from model path: {e}")
        print(f"Trying to load tokenizer from base model: {base_model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            print("Loaded tokenizer from base model")
        except Exception as e2:
            print(f"Failed to load tokenizer from base model: {e2}")
            raise e2
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    try:
        # Try to load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("Loaded model successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

def generate_batch(model, tokenizer, texts, batch_size=16, max_new_tokens=256):
    """Generate predictions in batches"""
    predictions = []
    device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating predictions"): # texts
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode outputs
            for j, output in enumerate(outputs):
                # Get only the new generated tokens
                input_length = inputs['input_ids'][j].shape[0]
                generated = output[input_length:]
                
                generated_text = tokenizer.decode(generated, skip_special_tokens=True)
                predictions.append(generated_text.strip())
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add fallback predictions for failed batch
            for _ in range(len(batch_texts)):
                predictions.append("Answer: Generation failed")
    
    return predictions

def check_model_structure(model_path):
    """Check the structure of the merged model directory"""
    print(f"Path: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model directory does not exist")
        return False
    
    files = os.listdir(model_path)
    print(f"Files: {files}")
    
    # Check for essential files
    essential_files = {
        'config.json': False,
        'pytorch_model': False,  # Can be .bin or .safetensors or sharded
        'tokenizer': False       # Any tokenizer file
    }
    
    for file in files:
        if file == 'config.json':
            essential_files['config.json'] = True
        elif file.startswith('pytorch_model') or file.endswith('.safetensors'):
            essential_files['pytorch_model'] = True
        elif 'tokenizer' in file:
            essential_files['tokenizer'] = True
    
    print("\nEssential files check:")
    all_good = True
    for file, found in essential_files.items():
        status = "ok" if found else "error"
        print(f"{status} {file}: {found}")
        if not found:
            all_good = False
    
    # Check config.json content
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Model type: {config.get('model_type', 'unknown')}")
            print(f"Architecture: {config.get('architectures', 'unknown')}")
        except Exception as e:
            print(f"Error reading config: {e}")
            all_good = False
    
    return all_good

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py input_dataset_path output_csv_path model_path")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = sys.argv[3]

    print("=== Simple Inference Script (Transformers-based) ===")
    
    # Check model structure
    model_ok = check_model_structure(model_path)
    if not model_ok:
        print("Model structure has issues, but attempting to continue...")
    
    # Load dataset
    print(f"\nLoading dataset from {input_path}")
    try:
        dataset = load_from_disk(input_path)
        print(f"Dataset loaded with {len(dataset)} examples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
    except Exception as e:
        print(f"Failed to load model and tokenizer: {e}")
        print("Creating dummy predictions...")
        
        # Create dummy predictions as fallback
        predictions = ["Answer: Model loading failed"] * len(dataset)
        
        results_df = pd.DataFrame({
            'id': dataset['id'] if 'id' in dataset.column_names else list(range(len(dataset))),
            'label': dataset['label'] if 'label' in dataset.column_names else [0] * len(dataset),
            'prediction_text': predictions
        })
        
        results_df.to_csv(output_path, index=False)
        print(f"Saved dummy predictions to {output_path}")
        return
    
    # Generate predictions
    print(f"\nGenerating predictions...")
    try:
        texts = dataset['text']
        predictions = generate_batch(model, tokenizer, texts, batch_size=2)  # Small batch size for safety
        print(f"Generated {len(predictions)} predictions")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        predictions = [f"Answer: Generation error - {str(e)[:50]}"] * len(dataset)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'id': dataset['id'] if 'id' in dataset.column_names else list(range(len(dataset))),
        'label': dataset['label'] if 'label' in dataset.column_names else [0] * len(dataset),
        'prediction_text': predictions
    })

    # Save results
    try:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")
        sys.exit(1)
    
    # Print sample predictions
    print(f"\nSample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"  {i+1}. {predictions[i][:100]}...")

if __name__ == "__main__":
    main()