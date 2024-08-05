import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, d_embed, d_query):
        super().__init__()
        self.query = nn.Linear(d_embed, d_query, bias=False)
        self.key = nn.Linear(d_embed, d_query, bias=False)
        self.value = nn.Linear(d_embed, d_query, bias=False)
        # TODO dropout

    def forward(self, x, attn_mask=None):
        """
        T: context_length, T stands for time, just a naming convention.
        Shapes:
            x      : (B, T, d_embed) 
            q, k, v: (B, T, d_query) <- (B, T, d_embed) @ (d_embed, d_query)
            logits : (B, T, T) <- (B, T, d_query) @ (B, d_query, T) 
            A      : (B, T, T)
            output : (B, T, d_query)
        """
        B, T, C = x.shape
        q: Tensor = self.query(x)
        k: Tensor = self.key(x)
        v: Tensor = self.value(x)
        # Compute attention scores
        logits = q @ k.transpose(1, 2) # a(q, k) for each cell. # TODO * k.shape[-1]**-0.5 
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, attn_mask.shape[1], 1)
            logits = logits.masked_fill(attn_mask.logical_not(), float(-1e9))
        A = F.softmax(logits, dim=-1)
        A = A.masked_fill(attn_mask.logical_not(), 0) # TODO
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
            x: (batch_size, ctx_len, d_embed)
        Return:
            out: (batch_size, ctx_len, d_embed)
        """
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1) # concat in channel, -> (B, T, C*num_heads)
        out = self.proj(out)
        # TODO out = self.dropout(out)
        return out