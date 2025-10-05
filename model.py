import torch
from torch import nn
from math import sqrt

""" Feed-Forward network from the transformer block """
class FF_network(nn.Module):
    # simple feed-forward network with one hidden layer
    def __init__(self, d, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff)
        self.W2 = nn.Linear(d_ff, d)

    def forward(self, x):
        x = self.W1(x)
        x = self.W2(x)
        return x

""" Multi-headed attention """
class Attention(nn.Module):
    def __init__(self, n_heads, d, d_k, d_v):
        super().__init__()
        # dimensional attributes
        self.n_heads = n_heads
        self.d = d
        self.d_k = d_k # dimensionality of W_k and W_q matrixes
        self.d_v = d_v # dimensionality of W_v matrix

        # learnable parameters (for each head)
        self.W_k = nn.ParameterList([torch.rand(d, d_k) for _ in range(n_heads)]) # key matrix
        self.W_q = nn.ParameterList([torch.rand(d, d_k) for _ in range(n_heads)]) # query matrix
        self.W_v = nn.ParameterList([torch.rand(d, d_v) for _ in range(n_heads)]) # value matrix
        self.W_0 = nn.Parameter(torch.rand(d_v*n_heads, d)) # W_0 matrix -> format result to dimesionality d

    def forward(self, x):
        heads_list = []
        for head in range(self.n_heads):
            # compute Q, K, V for the input
            Q = x @ self.W_q[head]
            K = x @ self.W_k[head]
            V = x @ self.W_v[head]

            # masked score
            score = (Q @ K.transpose(-2,-1)) / sqrt(self.d_k)
            mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
            score = score.masked_fill(mask, float('-inf'))

            # single-head attention computing
            attn = torch.softmax(score, dim=-1)
            head_i = attn @ V
            heads_list.append(head_i)

        # concatenate all attention heads and format with matrix W_0 into [d, d]
        heads = torch.cat(heads_list, dim=-1)
        A = heads @ self.W_0
        return A

""" Single transformer block """
class Transformer_Block(nn.Module):
    def __init__(self, n_heads, d, d_k, d_v, d_ff, dropout):
        super().__init__()
        # all layers
        self.ln1 = nn.LayerNorm(d)
        self.attention = Attention(n_heads, d, d_k, d_v)
        self.ln2 = nn.LayerNorm(d)
        self.ff = FF_network(d, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # forward data using a residual connection 
        h = x + self.dropout(self.attention.forward(self.ln1.forward(x)))
        o = h + self.dropout(self.ff.forward(self.ln2.forward(h)))
        return o

""" Transformer -> Sequence of transformer blocks """
class Transformer(nn.Module):
    def __init__(self, n_heads, n_blocks, d, d_k, d_v, d_ff, dropout):
        super().__init__()
        # list of all transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [Transformer_Block(n_heads, d, d_k, d_v, d_ff, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x):
        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x)
        return x

""" 
Actual model class
(inputs should be tokenized before)

Parameters:
max_seq_len - context window of the model (tokens)
vocab_size - size of the token vocabulary used
n_heads - number of attention heads
n_blocks - number of sequential transformer blocks
d_model - model dimensionality (embedding size)
d_k - dimensionality of the query and key matrices (W_q, W_k)
d_v - dimensionality of the value matrix (W_v)
d_ff - dimension of the hidden layer of the FF network
dropout - dropout value
"""
class LLM(nn.Module):
    def __init__(self, max_seq_len, vocab_size, n_heads, n_blocks, d_model, d_k, d_v, d_ff, dropout):
        super().__init__()
        # Learnable embeddings + (positional embeddings -> on AIAYN paper sinosoidal functions are used)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # transformer
        self.transformer = Transformer(n_heads, n_blocks, d_model, d_k, d_v, d_ff, dropout)

        # last linear layer
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # get embeddings
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)     

        # forward embeddings into the transformer
        h = self.transformer.forward(x)

        # get logit score prediction for the whole vocabulary
        logits = self.out(h)
        return logits
