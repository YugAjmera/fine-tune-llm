import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = torch.nn.Linear(in_dim, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)                           # Zero initialize B so LoRA has no initial effect
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (self.lora_B(self.lora_A(x)))


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)