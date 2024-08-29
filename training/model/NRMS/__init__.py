import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from config import BaseConfig
class NRMS(nn.Module):
    """
    NewsEncoder: (candidate_news) -> news_vector
    UserEncoder: (clicked_news) -> user_vector
    model      : (clicked_news, candidate_news) -> click_probability
    """
    def __init__(
            self,
            config: BaseConfig,
            pretrained_embedding=None):
        super().__init__()
        self.config = config
        self.device = config.device
        # ---- Layers ---- #
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(config.vocab_size,
                                          config.embedding_dim,
                                          padding_idx=0) # TODO padding_idx
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=False,
                                                          padding_idx=0)
        self.news_encoder = Encoder(config.num_heads, config.embedding_dim)
        self.user_encoder = Encoder(config.num_heads, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.to(self.device)
    
    def forward(self,
                clicked_news: dict,
                candidate_news: dict):
        """
        Args:
            title             : (batch_size, num_news, ctx_len)
            title_mask        : (batch_size, num_news, ctx_len)
            embed             : (batch_size, num_news, ctx_len, d_embed)
            clicked_news_vec  : (batch_size, num_clicked_news, d_embed)
            candidate_news_vec: (batch_size, num_candidate_news, d_embed)
        """
        # Clicked news
        embed = self.dropout(self.embedding(clicked_news['title'].to(self.device)))
        clicked_news_vec = self.news_encoder(embed, clicked_news['title_mask'].to(self.device))
        final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
        # Candidate news
        embed = self.embedding(candidate_news['title'].to(self.device))
        candidate_news_vec = self.news_encoder(embed, candidate_news['title_mask'].to(self.device))
        # Dot product
        click_probability = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        return click_probability
