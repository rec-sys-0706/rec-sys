import torch
import torch.nn as nn
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
            device):
        super().__init__()
        self.config = config
        self.device = device
        self.to(device)
        # ---- Layers ---- #
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.news_encoder = Encoder(config.num_heads, config.embedding_dim)
        self.user_encoder = Encoder(config.num_heads, config.embedding_dim)
    
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
        embed = self.embedding(clicked_news['title'].to(self.device))
        clicked_news_vec = self.news_encoder(embed, clicked_news['title_mask'].to(self.device))
        final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
        # Candidate news
        embed = self.embedding(candidate_news['title'].to(self.device))
        candidate_news_vec = self.news_encoder(embed, candidate_news['title_mask'].to(self.device))
        # Dot product
        click_probability = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        return click_probability
