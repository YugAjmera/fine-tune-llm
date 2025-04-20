# Code to parse the Alpaca-GPT-4 dataset
import torch
from torch.utils.data import Dataset

class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Format the data in alpaca style template & tokenize it
        full_text = self.format_alpaca_style(self.data[index])
        tokens = self.tokenizer.encode(full_text)
        return torch.tensor(tokens)

    @staticmethod
    def format_alpaca_style(entry, no_reponse=False):
        if len(entry['input']) == 0:
            prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            input_text = ""
        else:
            prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
            input_text = "\n\n### Input:\n" + entry['input']
        instruction_text = "\n\n### Instruction:\n" + entry['instruction']
        if no_reponse:
            response = ""
        else:
            response = "\n\n### Response:\n" + entry['output']
        prompt += instruction_text + input_text + response
        return prompt

    @staticmethod
    def collated_func(batch, pad_token_id=50256, ignore_index=-100, max_length=1024):
        # A custom collate function that pads each sequence to make them of same length.
        # This approach extends sequences to match the longest one in each batch, not the whole dataset. Hence, efficient!

        batch_max_len = max(len(item) for item in batch)         
        inputs, targets = [], []
        for item in batch:
            pad_len = batch_max_len - len(item)
            # Add the end of text token to signify that the generated response is completed.
            # Adding 1 also creates the block data which we use to create inputs and targets
            block_data = torch.cat([item, torch.tensor([pad_token_id])])                           
            input_tokens = torch.cat([block_data, torch.full((pad_len,), pad_token_id)])[:-1]
            # Targets are inputs shifted one position to the right. 
            # Instead of padding token, we use "-100" value to exclude them from training loss calculation.
            # Default setting of cross entropy loss uses -100 as ignore index. Ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            target_tokens =  torch.cat([block_data, torch.full((pad_len,), ignore_index)])[1:]  
            inputs.append(input_tokens[:max_length])
            targets.append(target_tokens[:max_length])   

            # TO DO: masking on the prompt if needed.
        return torch.stack(inputs), torch.stack(targets)
    