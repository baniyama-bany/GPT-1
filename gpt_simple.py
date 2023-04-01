import copy
import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def make_causal_mask(self, attention_logit):
        length = attention_logit.size(-2)
        mask = torch.triu(torch.ones(length, length), 1) * -10000000.0
        return mask.to(attention_logit.device)

    def forward(self, Q, K, V, mask=None):
        attention_logit = torch.einsum("bhld,bhmd -> bhlm", Q, K) / math.sqrt(self.config.head_num)
        
        if self.config.causal:
            attention_logit += self.make_causal_mask(attention_logit)[None, None, :, :]

        attention_weight = F.softmax(attention_logit, dim=-1)
        X = torch.einsum("bhlm, bhmd -> bhld", attention_weight, V)

        return X


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_q = nn.Linear(config.dim, config.dim)
        self.W_k = nn.Linear(config.dim, config.dim)
        self.W_v = nn.Linear(config.dim, config.dim)

        self.attn = SoftmaxAttention(config)

    def forward(self, X):
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        attn_out = self.attn(Q, K, V)
        attn_out = self.combine_heads(attn_out)
        return attn_out

    # [batch, head, len, head_dim] -> [batch, len, dim]
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.config.head_num * self.config.head_dim)
        return X

    # [batch, len, dim] -> [batch, head, len, head_dim]
    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.config.head_num, self.config.head_dim)
        X = X.transpose(1, 2)
        return X


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn1 = nn.Linear(config.dim, config.dim_ff)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(config.dim_ff, config.dim)
    
    def forward(self, x):
        x = self.ffn1(x)
        x = self.relu(x)
        x = self.ffn2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FFN(config)
        self.norm1 = nn.LayerNorm(config.dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.dim, eps=1e-5)
    
    def forward(self, x):
        x = self.attn(x) + x
        x = self.norm1(x)
        x = self.ffn(x) + x
        x = self.norm2(x)
        return x


class GPT1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for i in range(config.layers_n)])
        self.word_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):

        pos_ids = torch.arange(x.size(-1)).to(x.device)
        x = self.word_embedding(x)

        x = x + self.position_embedding(pos_ids)[None, :]

        for layer in self.layers:
            x = layer(x)
        
        logits = self.output(x)

        return logits


class GPT1Config:
    def __init__(
            self,
            dim = 128,
            dim_ff = 256,
            vocab_size = 1000,
            max_length = 128,
            layers_n = 3,
            head_num = 4,
            causal = True,
    ):
        self.dim = dim
        self.dim_ff = dim_ff
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.layers_n = layers_n
        self.head_num = head_num
        self.head_dim = self.dim // self.head_num
        self.causal = causal

# config = Config()

# model = Decoder(config)
# # logits = model(torch.tensor([[1,2,3,4,5]]))
# # logits.sum().backward()




