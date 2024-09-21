import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, d_embed, d_query):
        super().__init__()
        self.query = nn.Linear(d_embed, d_query, bias=False)
        self.key = nn.Linear(d_embed, d_query, bias=False)
        self.value = nn.Linear(d_embed, d_query, bias=False)
        # TODO dropout

    def forward(self, x, attn_mask: Tensor=None):
        """
        N: number of news.
        T: context_length, T stands for time, just a naming convention.
        Shapes:
            x      : (..., T, d_embed) 
            q, k, v: (..., T, d_query) <- (..., T, d_embed) @ (d_embed, d_query)
            logits : (..., T, T) <- (..., T, d_query) @ (B, d_query, T) 
            A      : (..., T, T)
            output : (..., T, d_query)
        """
        q: Tensor = self.query(x)
        k: Tensor = self.key(x)
        v: Tensor = self.value(x)
        # Compute attention scores
        logits = q @ k.transpose(-1, -2) # a(q, k) for each cell. # TODO * k.shape[-1]**-0.5 
        if attn_mask is not None:
            repeats = [1] * (attn_mask.dim() + 1)
            repeats[-2] = attn_mask.shape[-1]
            attn_mask = attn_mask.unsqueeze(-2).repeat(repeats=repeats) # (repeats=(1, 1, dim, 1))
            logits = logits.masked_fill(attn_mask.logical_not(), float(-1e9)) # Mask 0 to -inf
        A = F.softmax(logits, dim=-1)
        # A = A.masked_fill(attn_mask.logical_not(), 0) # TODO delete
        return A @ v # ? weighted-sum
class MultiHeadAttention(nn.Module):
    """
    Description:
        Multiple heads of self-attention in parallel
    Args:
        n_heads       : 8 in original paper(OP).
        d_embed       : dimension of embedding vector
        d_query       : dimension of query (and key) vector
        ctx_len       : input vector sequence length
    """
    def __init__(self, n_heads, d_embed, d_query):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_embed, d_query) for _ in range(n_heads)])
        self.proj = nn.Linear(d_embed, d_embed)
        # TODO dropout
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (batch_size, n_news, ctx_len, d_embed)
        Return:
            out: (batch_size, n_news, ctx_len, d_embed)
        """
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1) # concat in channel, -> (B, T, C*num_heads)
        out = self.proj(out)
        # TODO out = self.dropout(out)
        return out