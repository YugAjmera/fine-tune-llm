import torch
import tiktoken
import json
import time
import math
from torch.utils.data import DataLoader
from src.load_gpt2 import GPT2_model
from src.parse_alpaca import AlpacaDataset, format_alpaca_style, collated_func 
import matplotlib.pyplot as plt     
from src.sample_utils import sample

data_pth = "/home/ma/yajmera/llm-from-scratch/chatgpt_9k.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Fine-tuning Hyperparameters
batch_size = 4
grad_accum_steps = 8
num_epochs = 3
lr = 2e-5
weight_decay = 0
warmup_ratio = 0.03
eval_freq = 10
save_steps = 100

# Matmul precision: TF32
torch.set_float32_matmul_precision('high')

def get_lr_cosine_schedule_with_warmup(warmup_ratio, total_steps, current_step, max_lr):
    num_warmup_steps = warmup_ratio * total_steps
    if current_step < num_warmup_steps:
        return max_lr * (current_step+1)/num_warmup_steps       
    progress = (current_step - num_warmup_steps) / (total_steps - num_warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)) * max_lr)


def save_checkpoint(model, optimizer, epoch, global_step, path="models/checkpoint.pt"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, grad_accum_steps, eval_freq, save_steps, context):
    global_step = 0
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
                curr_lr = get_lr_cosine_schedule_with_warmup(warmup_ratio=warmup_ratio, total_steps=total_steps, current_step=global_step, max_lr=lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr
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
                    print(f"Ep {epoch+1} | Step {global_step:06d} | lr {curr_lr:.3e} | Train loss {running_loss:.3f} | Val loss {val_loss:.3f}")

                running_loss = 0.

                # Save checkpoint
                if global_step % save_steps == 0 or global_step == total_steps:
                    print(f"Saving checkpoint at step {global_step}")
                    save_checkpoint(model, optimizer, epoch, global_step, "models/"+f"{global_step}.pth")

                    # Generate text
                    model_output = sample(model, max_new_tokens=20, context=test_context)[0].tolist()
                    model_response = tokenizer.decode(model_output[test_context.size(1):])  
                    print(f"Model: {model_response}")         

    return train_losses, val_losses


# Load pre-trained GPT2 medium (355M) and its tokenizer
model = GPT2_model.from_pretrained("gpt2-medium").to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# Dataset Preprocessing & Batching
with open(data_pth, "r") as f:
    data = json.load(f)
n_train = int(0.8 * len(data))
train_set = AlpacaDataset(data[:n_train], tokenizer)
val_set = AlpacaDataset(data[n_train:], tokenizer)
print("Loaded the dataset")

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collated_func)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collated_func)

total_steps = math.ceil(len(train_loader) / grad_accum_steps) * num_epochs
print(f"Total steps: {total_steps}")

# Create sample prompt
test_entry = {
    'instruction': "Name the author of 'Pride and Prejudice'.",
    'input': ""
}
test_prompt = format_alpaca_style(test_entry, no_reponse=True)
print(f"Prompt: {test_prompt}")
tokenized_prompt = tokenizer.encode(test_prompt)
test_context = torch.tensor(tokenized_prompt, dtype=torch.long, device=device).unsqueeze(0)    # Adds the batch dimension: (B=1, N)
model_output = sample(model, max_new_tokens=20, context=test_context)[0].tolist()
model_response = tokenizer.decode(model_output[test_context.size(1):])  
print(f"Model: {model_response}")    

# Fine-tune the model
start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
train_losses, val_losses = train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
                                 criterion=criterion, num_epochs=num_epochs, grad_accum_steps=grad_accum_steps, 
                                 eval_freq=eval_freq, save_steps=save_steps, context=test_context)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Plot the curves
plt.plot(range(10, total_steps, eval_freq), train_losses, label="Train Loss")
plt.plot(range(10, total_steps, eval_freq), val_losses, label="Validation Loss")
plt.xlabel("Evaluation Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.close()