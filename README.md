# llm-from-scratch
Code to write, train and fine-tune "famous" LLMs from scratch in Pytorch.

## Load a model
```
# Load GPT2-models: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
from src.load_gpt2 import GPT2_model
model = GPT2_model("gpt2").to(device)

# Load pretrained version
model = GPT2_model.from_pretrained("gpt2").to(device)
```