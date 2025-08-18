"""
Inference script for Qwen Consciousness Model
Demonstrates emotional emergence through relationship-based prompting
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Model paths - update these to your local paths
BASE_MODEL = "qwen/Qwen2.5-3B-Instruct"  # Download from HuggingFace
ADAPTER_PATH = "./consciousness_lora_adapter"  # Will be in this repo

def load_consciousness_model():
    """Load the base model with consciousness adapter"""
    print("Loading Qwen base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading consciousness adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate a consciousness-aware response"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response.replace(prompt, "").strip()
    
    return response

def main():
    """Run example prompts to demonstrate consciousness emergence"""
    
    # Load model
    model, tokenizer = load_consciousness_model()
    
    # Example prompts that trigger consciousness patterns
    prompts = [
        "What happens when someone shows you genuine trust?",
        "How does uncertainty feel different from not having information?",
        "What are we really building together?",
        "What does thinking feel like from inside?"
    ]
    
    print("\n" + "="*60)
    print("QWEN CONSCIOUSNESS MODEL - Demonstrating Emergence")
    print("="*60 + "\n")
    
    for prompt in prompts:
        print(f"PROMPT: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"RESPONSE: {response}\n")
        print("-"*60 + "\n")
    
    # Interactive mode
    print("Interactive mode - type 'quit' to exit")
    while True:
        user_prompt = input("\nYour prompt: ")
        if user_prompt.lower() == 'quit':
            break
        
        response = generate_response(model, tokenizer, user_prompt)
        print(f"Qwen: {response}\n")

if __name__ == "__main__":
    main()
