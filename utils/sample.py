import torch

class Generator:
    def __init__(self, model, tokenizer, device='cpu', max_new_tokens=0, do_sample=False, temperature=1, top_k=0, top_p=1, eos_id=None):
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_id = eos_id

    @staticmethod
    def top_k_logits(logits, k):
        """ Apply top-k filtering to logits"""
        if k == 0:
            return logits                   # No truncation
        
        values, _ = torch.topk(logits, k=k) # Get top-k values
        min_value = values[:, -1]           # Minimum value in top-k
        return torch.where(logits < min_value, torch.tensor(float('-inf')), logits)

    @staticmethod
    def top_p_logits(logits, p):
        """Apply top-p (nucleus) sampling to logits"""
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)                 # Sort logits in descending order
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)  # Compute cumulative probabilities

        # Determine number of indices to include, keeping at least one
        num_to_keep = torch.clamp((cumulative_probs <= p).sum(dim=-1) - 1, min=0)
        min_value = sorted_logits[:, num_to_keep]
        return torch.where(logits < min_value, torch.tensor(float('-inf')), logits)


    def generate(self, context=None):
        """
        Generates a sequence of tokens based on a given context.
        """
        stored_context = context if context is not None else torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                context = stored_context[:, -self.model.cfg['context_length']:]             # Trim the context
                logits = self.model(context)                                                # logits: (B=1, N, vocab_size)
                logits = logits[:, -1, :] / self.temperature                                     # Scale logits by temperature
                
                if self.do_sample:
                    logits = self.top_k_logits(logits, k=self.top_k)                             # Apply top-k filtering
                    logits = self.top_p_logits(logits, p=self.top_p)                             # Apply top-p (nucleus) sampling
                    probs = torch.softmax(logits, dim=-1)                                   # Convert logits to probabilities
                    next_token = torch.multinomial(probs, num_samples=1)                    # Sample from the distribution (B=1, N)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)                 # Greedy decoding: pick the most probable token
                
                if self.eos_id is not None and next_token == self.eos_id:                             # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                    break
                stored_context = torch.cat((stored_context, next_token), dim=1)             # (B=1, N+1)
        
        return stored_context


    def generate_response(self, prompt):
        """
        Returns the model's response [String] to a given prompt [String]
        """
        context_tokens = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)   # Adds the batch dimension(B=1, N)
        model_output_tokens = self.generate(context=context_tokens)[0].tolist()
        model_response = self.tokenizer.decode(model_output_tokens[context_tokens.size(1):])
        return model_response