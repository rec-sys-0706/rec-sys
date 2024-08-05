import torch
import torch.nn as nn
from .news_encoder import NewsEncoder
from config import BaseConfig

class NRMS(nn.Module):
    """
    NewsEncoder: (candidate_news) -> news_vector
    UserEncoder: (clicked_news) -> user_vector
    model      : (clicked_news, candidate_news) -> click_probability
    """
    def __init__(
            self,
            config: BaseConfig):
        super().__init__()
        self.news_encoder = NewsEncoder(config.vocab_size,
                                        config.dim_embedding,
                                        config.num_heads)
        # self.user_encoder = NewsEncoder()
    
    def forward(self,
                clicked_news: dict,
                candidate_news: list[dict]):
        # dict is {
        # news: (batch_size, ctx_len)
        # mask: (batch_size, ctx_len)
        # }

        # TODO stacking!
        print(candidate_news['title'].shape, len(clicked_news), clicked_news[0]['title'].shape)
        # candidate_news_vec = self.news_encoder(candidate_news)
        # clicked_news_vec = self.news_encoder(clicked_news)
        return torch.rand(candidate_news['title'].shape[0])
    
        # return click_probability
