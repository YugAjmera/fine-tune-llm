# Code to generate text from a model give the sampling hyperparameters 
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
def top_k_logits(logits, k):
    # Top-k sampling
    if k == 0:
        return logits                   # No truncation
    
    values, _ = torch.topk(logits, k=k) # Get top-k values
    min_value = values[:, -1]           # Minimum value in top-k
    return torch.where(logits < min_value, torch.tensor(float('-inf')), logits)


def top_p_logits(logits, p):
    # Nucleus Sampling
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)                 # Sort logits in descending order
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)  # Compute cumulative probabilities

    # Determine number of indices to include, keeping at least one
    num_to_keep = torch.clamp((cumulative_probs <= p).sum(dim=-1) - 1, min=0)
    min_value = sorted_logits[:, num_to_keep]
    return torch.where(logits < min_value, torch.tensor(float('-inf')), logits)


def sample(model, max_new_tokens, context=None, do_sample=False, temperature=1, top_k=0, top_p=1, eos_id=50256):
    stored_context = context
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            context = stored_context[:, -model.cfg['context_length']:]             # Trim the context
            logits = model(context)                                                # logits: (B=1, N, vocab_size)
            logits = logits[:, -1, :] / temperature                                # Scale logits by temperature
            if do_sample:
                logits = top_k_logits(logits, k=top_k)                             # Apply top-k filtering
                logits = top_p_logits(logits, p=top_p)                             # Apply top-p (nucleus) sampling
                probs = torch.softmax(logits, dim=-1)                              # Convert logits to probabilities
                next_token = torch.multinomial(probs, num_samples=1)               # Sample from the distribution (B=1, N)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if next_token == eos_id:                                               # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break
            stored_context = torch.cat((stored_context, next_token), dim=1)        # (B=1, N+1)
    return stored_context