import torch
import tiktoken
from utils.load_gpt2 import GPT2_model
from utils.alpaca_dataset import format_alpaca_style
from utils.sample import Generator

checkpoint_pth = "/home/ma/yajmera/llm-from-scratch/saved_checkpoints/693.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize GPT2 medium (355M) and its tokenizer
print("Loading model...")
model = GPT2_model("gpt2-medium").to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# Load the checkpoints weights in the model
checkpoint = torch.load(checkpoint_pth, map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"])

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
                      max_new_tokens=100, 
                      do_sample=True,
                      temperature=0.7,
                      top_p=0.9,
                      top_k=0,
                      eos_id=50256)
        model_response = generator.generate_response(prompt=prompt) 
        print(f"Model: {model_response}\n")    
