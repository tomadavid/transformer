import torch
from torch.utils.data import Dataset

""" 
Builds a tokenized dataset out of a text corpus
(uses only one special token <EOS> (end-of-sequence)
"""
class TextDataset(Dataset):
    def __init__(self, corpus, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokens = []

        for text in corpus:
            # Encode text using BPE tokenizer
            encoded = tokenizer.encode(text).ids  # get list of token IDs
            # Append EOS token
            encoded.append(tokenizer.token_to_id("<EOS>"))
            self.tokens.extend(encoded)

    def __len__(self):
        # returns number of possible sequences
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        # get sequence of data from index idx and its targets
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)
