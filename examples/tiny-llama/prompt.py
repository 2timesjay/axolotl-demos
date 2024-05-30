import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel, PeftConfig  # https://github.com/huggingface/peft

def main(use_adapter):
    base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    adapter_path = "./outputs/tiny-llama-lora/"

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if use_adapter:
        # Load the adapter configuration
        peft_config = PeftConfig.from_pretrained(adapter_path)

        # Apply the LoRA adapter to the base model
        model = PeftModel.from_pretrained(model, adapter_path, config=peft_config)

    # Now you can use the model with or without the adapter applied
    input_text = "Give three tips for staying healthy."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate text using a language model.")
    parser.add_argument("--use-adapter", action="store_true", help="Whether to use the adapter or not")
    args = parser.parse_args()
    main(args.use_adapter)