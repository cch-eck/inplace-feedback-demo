# model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Transformers version:", torch.__version__)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading model {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_response(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7):
    """Generate a single-turn chat completion."""
    # Qwen-style chat prompt (system + user)
    formatted = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # remove the prompt portion
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text

