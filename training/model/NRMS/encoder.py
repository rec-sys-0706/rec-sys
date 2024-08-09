import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general import AdditiveAttention, MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
        d_query = d_embed // n_heads # TODO There could be other ways.
        self.multi_head = MultiHeadAttention(n_heads, d_embed, d_query)
        self.additive = AdditiveAttention(d_embed, d_query)
        self.d_embed = d_embed
    def forward(self, embedding, attn_mask=None):
        """
        Args:
            embedding  : (batch_size, n_news, context_length, d_embed)
            attn_mask  : (batch_size, n_news, context_length)
        Return:
            news_vector: (batch_size, n_news, d_embed)
        """
        representations = self.multi_head(embedding, attn_mask)
        
        # TODO mask output vector and loss?
        # mask = attn_mask.unsqueeze(-1).repeat(1, 1, self.d_embed)
        # representations = representations.masked_fill(mask.logical_not(), 0)
        
        news_vector = self.additive(representations)
        return news_vector