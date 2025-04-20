import torch
import tiktoken
import json
from torch.utils.data import DataLoader
from src.load_gpt2 import GPT2_model
from src.parse_alpaca import AlpacaDataset        

data_pth = "/home/ma/yajmera/llm-from-scratch/alpaca_gpt4_data.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_eval_loss(model, loader, criterion, num_batches=None):
    model.eval()
    total_loss = 0.
    num_batches = len(loader) if num_batches is None else num_batches
    with torch.no_grad():
        for i, (context_batch, target_batch) in enumerate(loader):
            context_batch, target_batch = context_batch.to(device), target_batch.to(device)
            logits = model(context_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            total_loss += loss
            if i > num_batches:
                break
    return (total_loss/num_batches)


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, eval_freq, eval_iter):
    global_step = 0
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for context_batch, target_batch in train_loader:
            context_batch, target_batch = context_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            logits = model(context_batch)                                                       # Shape: (B, N, vocab_size)
            loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            global_step += 1
        
            # Evaluate the loss on train and val
            if global_step % eval_freq == 0:
                train_loss = get_eval_loss(model=model, loader=train_loader, criterion=criterion, num_batches=eval_iter)
                val_loss = get_eval_loss(model=model, loader=val_loader, criterion=criterion, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1}/{num_epochs} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

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

# Create data loaders
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=train_set.collated_func)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=val_set.collated_func)

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.1)
criterion = torch.nn.CrossEntropyLoss()
train_losses, val_losses = train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
                                 criterion=criterion, num_epochs=2, eval_freq=10, eval_iter=100)


