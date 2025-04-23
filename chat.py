import torch
import tiktoken
from utils.load_gpt2 import GPT2_model
from utils.alpaca_dataset import format_alpaca_style
from utils.sample import Generator

checkpoint_pth = "/home/ma/yajmera/llm-from-scratch/saved_checkpoints/lora_600.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Generation hyperparameters
max_new_tokens = 500
do_sample = True
temperature = 0.9
top_k = 40
top_p = 0.9
eos_id = 50256
use_lora = True
lora_r = 16
lora_alpha = 16

# Initialize GPT2 medium (355M) and its tokenizer
print("Loading model...")
tokenizer = tiktoken.get_encoding("gpt2")
if use_lora:
    model = GPT2_model.from_pretrained("gpt2-medium", use_lora=use_lora, lora_rank=lora_r, lora_alpha=lora_alpha).to(device)
else:
    model = GPT2_model("gpt2-medium").to(device)

# Load the checkpoints weights in the model
checkpoint = torch.load(checkpoint_pth, map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

print("🧠 Model is ready. Type your message and hit Enter. Type 'quit' to exit.\n")

while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        # Structure the user input in alpaca prompt-style
        prompt = format_alpaca_style({
            'instruction': user_input,
            'input': "",
            'output': ""
            })
        generator = Generator(model=model, 
                      tokenizer=tokenizer, 
                      device=device, 
                      max_new_tokens=max_new_tokens, 
                      do_sample=do_sample,
                      temperature=temperature,
                      top_k=top_k,
                      top_p=top_p,
                      eos_id=eos_id)
        model_response = generator.generate_response(prompt=prompt) 
        print(f"Model: {model_response}\n")    
