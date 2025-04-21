import torch
from torch.optim.lr_scheduler import _LRScheduler
from utils.sample import Generator
import math

class CosineWithWarmupScheduler(_LRScheduler):
    """
    Custom learning rate scheduler based on a cosine schedule with warmup.
    """
    def __init__(self, optimizer, max_lr, warmup_ratio, total_steps, last_epoch=-1):
        self.base_lr = max_lr
        self.num_warmup_steps = int(warmup_ratio * total_steps)
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.base_lr * (self.last_epoch + 1) / self.num_warmup_steps]
        else:
            progress = (self.last_epoch - self.num_warmup_steps) / max(1, self.total_steps - self.num_warmup_steps)
            return [max(0.0, self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))]
    

def train(model, device, train_loader, val_loader, optimizers: tuple, criterion, num_train_epochs, eval_freq, save_steps, grad_accum_steps=1, prompt=None, generator: Generator = None):
    """
    Trains the model with mixed precision, gradient accumulation and learning rate scheduling.
    """
    torch.set_float32_matmul_precision('high')
    optimizer, scheduler = optimizers
    global_step = 0
    train_losses, val_losses = [], []

    for epoch in range(num_train_epochs):
        
        # Training loop
        model.train()
        running_loss = 0.
        for step, (context_batch, target_batch) in enumerate(train_loader):
            context_batch, target_batch = context_batch.to(device), target_batch.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(context_batch)                                                       
                loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            
            loss = loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step() if scheduler else None
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
                    curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                    
                    print(f"Ep {epoch+1} | Step {global_step:06d} | lr {curr_lr:.3e} | Train loss {running_loss:.3f} | Val loss {val_loss:.3f}")

                running_loss = 0.

                # Save checkpoint
                if global_step % save_steps == 0 or (step == len(train_loader) - 1 and epoch == num_train_epochs - 1):
                    print(f"Saving checkpoint at step {global_step}")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "epoch": epoch,
                        "global_step": global_step,
                    }, "saved_checkpoints/"+f"{global_step}.pth")

                    # Generate text
                    if generator is not None:
                        model_response = generator.generate_response(prompt=prompt) 
                        print(f"Model: {model_response}")         

    return train_losses, val_losses