import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # Initialize
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == self.embed_size), "Embedding sizes need to be divisible by heads"

        # Initialize key, value and queries for self attention
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split embedding into different heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energie = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        #queries shape: (N, query_len, heads, head_dim)
        #keys shape: (N, key_len, heads, head_dim)
        #energie shape: (N, heads, query_len, key_len)

        # applying mask
        if mask is not None:
            energie = energie.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energie / (self.embed_size ** (1/2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # out shape: (N, query_len,heads, head_dim) and last 2 dimensions flatten
        # values shape: (N, value_len, heads, head_dim)
        out = self.fc_out(out)
        # (N, query_len, embed_size)
        return out

        


