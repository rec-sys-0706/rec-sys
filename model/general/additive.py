import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Description:
        Second-level attention for gathering attention representation infomation.
    Args:
        d_embed: dimension of input vector
        d_query: dimension of query vector
    """
    def __init__(self, d_embed, d_query):
        super().__init__()
        self.linear = nn.Linear(d_embed, d_query)
        self.query_vector = nn.Parameter(torch.randn(d_query) * 0.1)
    def forward(self, x: Tensor):
        """
        Args:
            x           : (batch_size, ctx_len, d_embed)
        Returns:
            second-level attention weight.
        Others:
            query_vector: (d_query)
            temp        : (batch_size, ctx_len, d_query)
            a           : (batch_size, ctx_len)
            out         : (batch_size, d_embed, 1)
        """
        temp = torch.tanh(self.linear(x))
        a = F.softmax(temp @ self.query_vector, dim=1).unsqueeze(dim=-1)
        out = x * a # weighted-sum # TODO? torch.bmm(x.transpose(1, 2), a)
        return out