from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py lora_adapter_path merged_model_path")
        sys.exit(1)

    # Extract command-line arguments
    lora_path = sys.argv[1]
    merged_model_path = sys.argv[2]
    
    # Load the model and merge it with lora adapter
    model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", use_cache=True, torch_dtype=torch.float16).to("cuda")
    lora_model = PeftModel.from_pretrained(model, lora_path)
    lora_model = lora_model.merge_and_unload()
    
    # Save merged model for the vLLM inference
    lora_model.save_pretrained(merged_model_path)
    
    