import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general import AdditiveAttention, MultiHeadAttention
import pdb
class NewsEncoder(nn.Module):
    def __init__(
            self,
            n_vocab,
            d_embed,
            n_heads):
        super().__init__()
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
        d_query = d_embed // n_heads # TODO There could be other ways.
        self.embedding = nn.Embedding(n_vocab, d_embed)
        self.multi_head = MultiHeadAttention(n_heads, d_embed, d_query)
        self.additive = AdditiveAttention(d_embed, d_query)
        self.d_embed = d_embed
    def forward(self, news, attn_mask=None):
        """
        Args:
            news : (batch_size, context_length, d_embed)
            embed: (batch_size, context_length, d_embed)
        """
        embd = self.embedding(news)
        representations = self.multi_head(embd, attn_mask)
        
        # TODO mask output vector? 
        mask = attn_mask.unsqueeze(-1).repeat(1, 1, self.d_embed)
        representations = representations.masked_fill(mask.logical_not(), 0)

        news_vector = self.additive(representations)
        return news_vector