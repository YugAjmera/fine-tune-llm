import torch
import tiktoken
import json
import time
import math
from torch.utils.data import DataLoader
from src.load_gpt2 import GPT2_model
from src.parse_alpaca import AlpacaDataset   
import matplotlib.pyplot as plt     

data_pth = "/home/ma/yajmera/llm-from-scratch/chatgpt_9k.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Fine-tuning Hyperparameters
batch_size = 8
grad_accum_steps = 4
num_epochs = 3
lr = 2e-5
weight_decay = 0

# Matmul precision: TF32
torch.set_float32_matmul_precision('high')

def save_checkpoint(model, optimizer, epoch, global_step, path="models/checkpoint.pt"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, grad_accum_steps, eval_freq, save_steps):
    global_step = -1
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.
        for step, (context_batch, target_batch) in enumerate(train_loader):
            context_batch, target_batch = context_batch.to(device), target_batch.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(context_batch)                                                       # Shape: (B, N, vocab_size)
                loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            
            loss = loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
                # Evaluate the model
                if global_step % eval_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.
                        for context_batch, target_batch in val_loader:
                            context_batch, target_batch = context_batch.to(device), target_batch.to(device)
                            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                logits = model(context_batch) 
                                loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
                            val_loss += loss.item() / len(val_loader)
                    
                    train_losses.append(running_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} | Step {global_step:06d} | Train loss {running_loss:.3f} | Val loss {val_loss:.3f}")

                running_loss = 0.

                # Save checkpoint
                if (global_step+1) % save_steps == 0:
                    print(f"Saving checkpoint at step {global_step}")
                    save_checkpoint(model, optimizer, epoch, global_step, "models/"+f"{global_step}.pth")

    return train_losses, val_losses


# Load pre-trained GPT2 medium (355M) and its tokenizer
model = GPT2_model.from_pretrained("gpt2").to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# Dataset Preprocessing & Batching
with open(data_pth, "r") as f:
    data = json.load(f)
n_train = int(0.8 * len(data))
train_set = AlpacaDataset(data[:n_train], tokenizer)
val_set = AlpacaDataset(data[n_train:], tokenizer)
print("Datasets loaded")

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_set.collated_func)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=val_set.collated_func)
print(f"1 epoch = {math.ceil(len(train_loader) / grad_accum_steps)} steps")
print(f"Total steps: {math.ceil(len(train_loader) / grad_accum_steps) * num_epochs}")

# Fine-tune the model
start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
train_losses, val_losses = train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
                                 criterion=criterion, num_epochs=num_epochs, grad_accum_steps=grad_accum_steps, 
                                 eval_freq=10, save_steps=200)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Plot the curves
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Evaluation Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.savefig("loss_plot.png")
plt.close()