import torch
import tiktoken
import json
import time
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from utils.load_gpt2 import GPT2_model
from utils.alpaca_dataset import AlpacaDataset, format_alpaca_style, collated_func 
from utils.sample import Generator
from utils.train import CosineWithWarmupScheduler, train

data_path = "/home/ma/yajmera/llm-from-scratch/datasets/chatgpt_9k.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Fine-tuning Hyperparameters
batch_size = 4
grad_accum_steps = 8
num_epochs = 3
lr = 5e-4
weight_decay = 0
warmup_ratio = 0.1
eval_freq = 10
save_steps = 200
use_lora = True
lora_r = 16
lora_alpha = 16

# Load pre-trained GPT2 medium (355M) and its tokenizer
print("Loading model...")
model = GPT2_model.from_pretrained("gpt2-medium", use_lora=use_lora, lora_rank=lora_r, lora_alpha=lora_alpha).to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# Dataset Preprocessing & Batching
print("Loading data...")
with open(data_path, "r") as f:
    data = json.load(f)
n_train = int(0.8 * len(data))
train_set = AlpacaDataset(data[:n_train], tokenizer)
val_set = AlpacaDataset(data[n_train:], tokenizer)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collated_func)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collated_func)

total_train_steps = math.ceil(len(train_loader) / grad_accum_steps) * num_epochs
print(f"Total training steps: {total_train_steps}")

# Create sample prompt & check initial output of the model
prompt = format_alpaca_style({
    'instruction': "Name the author of 'Pride and Prejudice'.",
    'input': "",
    'output': ""
    })
print(f"Sample prompt:\n{prompt}")
generator = Generator(model=model, 
                      tokenizer=tokenizer, 
                      device=device, 
                      max_new_tokens=25, 
                      do_sample=False,
                      eos_id=50256)
model_response = generator.generate_response(prompt=prompt) 
print(f"Model: {model_response}")    

# Fine-tune the model
start_time = time.time()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineWithWarmupScheduler(optimizer=optimizer, max_lr=lr, warmup_ratio=warmup_ratio, total_steps=total_train_steps)

train_losses, val_losses = train(model=model,
                                 device=device,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 optimizers=(optimizer, scheduler),
                                 criterion=criterion,
                                 num_train_epochs=num_epochs,
                                 eval_freq=eval_freq,
                                 save_steps=save_steps,
                                 grad_accum_steps=grad_accum_steps,
                                 prompt=prompt,
                                 generator=generator,
                                 save_path="saved_checkpoints/lora",
                                 use_lora=True)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Plot the curves
plt.plot(range(10, total_train_steps, eval_freq), train_losses, label="Train Loss")
plt.plot(range(10, total_train_steps, eval_freq), val_losses, label="Validation Loss")
plt.xlabel("Evaluation Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.close()
print("Training plots are saved")