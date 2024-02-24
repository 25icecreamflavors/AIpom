import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys

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
    ]

    return {"prompt_text": chat}


if __name__ == "__main__":
    # Check if all command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file.json output.csv checkpoint_path")
        sys.exit(1)

    # Extract command-line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_path = sys.argv[3]
    
    
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=1024)

    # Loading data
    test_dataset = load_dataset(
        "json",
        data_files=input_file,
        split="train",
        num_proc=16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Open-Orca/Mistral-7B-OpenOrca",
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = "<PAD>"

    # Preprocess the data
    test_dataset = test_dataset.map(create_instruct, num_proc=16)
    test_dataset = test_dataset.map(
        lambda x: {
            "prompt_chat": tokenizer.apply_chat_template(
                x["prompt_text"], tokenize=False, add_generation_prompt=True
            )
        }
    )

    outputs = llm.generate(test_dataset["prompt_chat"], sampling_params)

    answers_list = []
    # Collect the outputs
    for output in outputs:
        generated_text = output.outputs[0].text
        answers_list.append(generated_text)

    df_test = pd.DataFrame(
        test_dataset.remove_columns(
            [
                "text",
                "prompt_text",
                "prompt_chat",
            ]
        )
    )
    df_test["prediction_text"] = ""
    df_test["prediction_text"] = answers_list

    df_test.to_csv(output_file, index=False)
