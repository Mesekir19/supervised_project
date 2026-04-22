import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B"
local_model_path = "./qwen-poi-finetuned"

print("Loading base model...")

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float32
)

# 3. Loading custom tokenizer and applying on fine-tuned weights
print("Applying fine-tuned weights...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = PeftModel.from_pretrained(base_model, local_model_path)

print("Model loaded successfully! Ready for inference.\\n")
print("-" * 50)

def ask_my_model(question):
    prompt = f"<|user|> {question} <|assistant|>"
    
    # Convert text to tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the output back into human-readable text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip away the prompt part so we just see the answer
    answer = result.split("<|assistant|>")[-1].strip()
    return answer


question = "Where is the Centre Commercial Saint Sébastien relative to the Marché Central?"
print(f"Question: {question}")

answer = ask_my_model(question)
print(f"Answer: {answer}")