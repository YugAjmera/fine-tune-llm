import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import GPT2LMHeadModel
from models.gpt2 import GPT2

class GPT2_model(GPT2):
    def __init__(self, model_type="gpt2", cfg=None, verbose=False):
        default_config = {
            'gpt2':         dict(emb_dim=768,  n_layers=12, n_heads=12), # 124M params
            'gpt2-medium':  dict(emb_dim=1024, n_layers=24, n_heads=16), # 350M params
            'gpt2-large':   dict(emb_dim=1280, n_layers=36, n_heads=20), # 774M params
            'gpt2-xl':      dict(emb_dim=1600, n_layers=48, n_heads=25), # 1558M params
        }[model_type]
        default_config.update({
            "vocab_size": 50257,    # 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
            "context_length": 1024, # Maximum sequence length
            "attn_pdrop": 0.1,      # Dropout probability for attention dropout
            "embd_pdrop": 0.1,      # Dropout probability for embeddings dropout
            "resid_pdrop": 0.1,     # Dropout probability for residual dropout
            "qkv_bias": True        # Whether to use bias in QKV layer
        })
        # Merge user-defined config into default config; user values overwrites duplicates
        if cfg is not None:
            default_config.update(cfg)

        self.verbose = verbose
        if self.verbose:
            print(f"Configuration dictionary: {default_config}")

        # Initialize the GPT2 model with config
        super().__init__(default_config)
        print(f"Initialized {model_type} with {(sum(p.numel() for p in self.parameters()) / 1e6):.2f} M parameters.")
        pass
    
    @classmethod
    def from_pretrained(cls, model_type="gpt2", cfg=None, verbose=False):
        model = cls(model_type=model_type, cfg=cfg, verbose=verbose)
        current_params = model.state_dict()

        hf_model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
        hf_model_params = hf_model.state_dict()

        # OpenAI’s GPT-2 checkpoints use a Conv1d module on each linear layer.
        # As a result, the layers [c_attn, c_proj, c_fc and c_proj] need to be handled differently: transpose the weights before copying
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        with torch.no_grad():
            for name, param in hf_model_params.items():
                if name in current_params:                      # check if the parameter name matches
                    if name.endswith(tuple(transposed)):        # if the parameter has to be transposed
                        current_params[name].copy_(param.t())   # Transpose the weights and then copy
                    elif param.shape == current_params[name].shape:
                        current_params[name].copy_(param)       # copy the weights over
                    else:
                        raise ValueError(f"Shape mismatch for parameter '{name}': {param.shape} (HF) vs {current_params[name].shape} (Model)")
                else:
                    raise KeyError(f"Parameter '{name}' not found in your model")
        if verbose:    
            print("Loaded weights from OpenAI checkpoint successfully!")
        return model






