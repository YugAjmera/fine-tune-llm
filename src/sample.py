import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sample:
    def __init__(self, model):
        self.model = model.to(device)
        self.model.eval()  

    @staticmethod
    def top_k_logits(logits, k):
        # Top-k sampling
        if k == 0:
            return logits                   # No truncation
        
        values, _ = torch.topk(logits, k=k) # Get top-k values
        min_value = values[:, -1]           # Minimum value in top-k
        return torch.where(logits < min_value, torch.tensor(float('-inf')), logits)

    @staticmethod
    def top_p_logits(logits, p):
        # Nucleus Sampling
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)                 # Sort logits in descending order
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)  # Compute cumulative probabilities

        # Determine number of indices to include, keeping at least one
        num_to_keep = torch.clamp((cumulative_probs <= p).sum(dim=-1) - 1, min=0)
        min_value = sorted_logits[:, num_to_keep]
        return torch.where(logits < min_value, torch.tensor(float('-inf')), logits)


    def generate(self, max_new_tokens, start_token=None, context=None, temperature=1, top_k=0, top_p=1):
        if start_token is None:
            assert context is not None, 'Specify exactly one of start_token and context!'
        else:
            assert context is None, 'Specify exactly one of start_token and context!'
            context = torch.full((1, 1), start_token, dtype=torch.long, device=device)
        
        stored_context = context

        for _ in range(max_new_tokens):
            with torch.no_grad():
                context = stored_context[:, -self.model.cfg['context_length']:]
                logits = self.model(context)                                       # logits: (B=1, N, vocab_size)
                logits = logits[:, -1, :] / temperature                            # Scale logits by temperature
                logits = self.top_k_logits(logits, k=top_k)                        # Apply top-k filtering
                logits = self.top_p_logits(logits, p=top_p)                        # Apply top-p (nucleus) sampling
                probs = torch.softmax(logits, dim=-1)                              # Convert logits to probabilities
                next_token = torch.multinomial(probs, num_samples=1)               # Sample from the distribution (B=1, N)
                stored_context = torch.cat((stored_context, next_token), dim=1)    # (B=1, N+1)
        
        return stored_context