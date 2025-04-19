import torch
import tiktoken
import json
from torch.utils.data import DataLoader
from src.load_gpt2 import GPT2_model
from src.parse_alpaca import AlpacaDataset
from src.sample import Sample        

data_pth = "/home/ma/yajmera/llm-from-scratch/alpaca_gpt4_data.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained GPT2 medium (355M) and its tokenizer
model = GPT2_model.from_pretrained("gpt2-medium").to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# Dataset Preprocessing & Batching
with open(data_pth, "r") as f:
    data = json.load(f)
n_train = int(0.8 * len(data))
train_set = AlpacaDataset(data[:n_train], tokenizer)
val_set = AlpacaDataset(data[n_train:], tokenizer)

# Create data loaders
train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=train_set.collated_func)
val_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=val_set.collated_func)

# Generate text
sample = Sample(model)
prompt = val_set.format_alpaca_style(val_set.data[2], no_reponse=True)
print(f"Prompt:{prompt}")
tokens = tokenizer.encode(prompt)
context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)    # Adds the batch dimension: (B=1, N)
output_tokens = sample.generate(context=context, max_new_tokens=50)
print("model generation:")
print(tokenizer.decode(output_tokens[0].tolist()))

