import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import math

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, attn_pdrop, num_heads, qkv_bias=False):
        super().__init__()
        self.c_attn = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.c_proj = nn.Linear(d_out, d_out)
        self.attn_dropout = nn.Dropout(attn_pdrop)

        self.d_h = d_out // num_heads                    # head dimension
        self.num_heads = num_heads
        
        self.register_buffer("masked", torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length))
    
    def forward(self, x):
        # Input vectors x - (B, N, d_in)
        B, N, d_in = x.shape
        # Obtain keys, values and queries in one go - (B, N, d_out)
        qkv = self.c_attn(x)
        queries, keys, values = qkv.chunk(3, dim=-1)
        
        # Split into H heads - (B, N, H, d_h) and then transpose to (B, H, N, d_h)
        k = keys.view(B, N, self.num_heads, self.d_h).transpose(1, 2)
        v = values.view(B, N, self.num_heads, self.d_h).transpose(1, 2)
        q = queries.view(B, N, self.num_heads, self.d_h).transpose(1, 2) 
        
        # Apply scaled dot-product attention with causal mask on each head
        attn_scores = (q @ k.transpose(2, 3)) * k.shape[-1]**-0.5    
        attn_scores = attn_scores.masked_fill(self.masked[:, :, :N, :N] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)                             
        attn_weights = self.attn_dropout(attn_weights)
        context_vec = attn_weights @ v

        # Concatenate: transpose back to (B, N, H, d_h), then combine heads (B, N, d_out)
        out = context_vec.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.c_proj(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)       # Used biased variance estimation (without Bessel's correction)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * norm_x + self.bias


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        )) 


class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, 4 * emb_dim)
        self.c_proj = nn.Linear(4 * emb_dim, emb_dim)
        self.gelu = GELU()

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(emb_dim=cfg['emb_dim'])
        self.attn = MultiHeadCausalAttention(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], context_length=cfg['context_length'], attn_pdrop=cfg['attn_pdrop'], num_heads=cfg['n_heads'], qkv_bias=cfg['qkv_bias'])
        self.ln_2 = LayerNorm(emb_dim=cfg['emb_dim'])
        self.mlp = MLP(emb_dim=cfg['emb_dim'])
        self.resid_dropout = nn.Dropout(cfg['resid_pdrop'])

    def forward(self, x):
        x = x + self.resid_dropout(self.attn(self.ln_1(x)))
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg['vocab_size'], cfg['emb_dim']),                      # Token Embeddings
            wpe = nn.Embedding(cfg['context_length'], cfg['emb_dim']),                  # Position Encoding
            embd_dropout = nn.Dropout(cfg['embd_pdrop']),                               # Embedding dropout
            h = nn.Sequential(*[DecoderBlock(cfg) for _ in range(cfg['n_layers'])]),    # Multiple Decoder blocks
            ln_f = LayerNorm(cfg['emb_dim']),                                           # Final layernorm
        ))
        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)         # Language modelling head

        self.transformer.wte.weight = self.lm_head.weight                               # Weight tying
        self.apply(self.init_weights)                                                   # Apply initialization

    def forward(self, x):
        B, N = x.size()
        token_emb = self.transformer.wte(x)                                             # (B, N, D)
        pos_emb = self.transformer.wpe(torch.arange(N, device=device))                  # (N, D)
        x = token_emb + pos_emb                                                         # (B, N, D)
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                                         # (B, N, vocab_size)
        return logits

    def init_weights(self, module):
        # Initialize weights as per GPT-2 huggingface code
        # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
        if isinstance(module, nn.Linear):        # Applies to linear layers only
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)      # Bias initialized to zero
        elif isinstance(module, nn.Embedding):   # Applied to embedding layer only
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Add residual scaling as per GPT-2 paper (There are 2 residual layers per Transformer Block)
        for name, param in module.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=(0.02/math.sqrt(2 * self.cfg['n_layers'])))