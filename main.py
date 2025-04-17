import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from src.load_gpt2 import GPT2_model
from src.sample import Sample
import tiktoken

# # Load pre-trained version
# model = GPT2_model.from_pretrained("gpt2").to(device)

# # Tokenize the input text
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello, I'm a language model,")
# prompt = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)    # Adds the batch dimension: (B=1, N)

# # Generate text
# sample = Sample(model)
# next_tokens = sample.generate(context=prompt, max_new_tokens=20)
# print(enc.decode(next_tokens[0].tolist()))


# Parse through the Alpaca-GPT-4 dataset
import json

with open("alpaca_gpt4_data.json", "r") as f:
    data = json.load(f)

print(len(data))
print(data[0])  # Example instruction