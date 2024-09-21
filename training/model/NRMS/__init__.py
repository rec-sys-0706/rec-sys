import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .encoder import Encoder

from parameters import Arguments

class NRMS(nn.Module):
    """
    NewsEncoder: (candidate_news) -> news_vector
    UserEncoder: (clicked_news) -> user_vector
    model      : (clicked_news, candidate_news) -> click_probability
    """
    def __init__(self, args: Arguments, vocab_size: int) -> None:
        super().__init__()
        self.args = args
        self.device = args.device
        # ---- Layers ---- #
        self.embedding = nn.Embedding(vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0) # TODO padding_idx?
        self.news_encoder = Encoder(args.num_heads, args.embedding_dim)
        self.user_encoder = Encoder(args.num_heads, args.embedding_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.to(self.device) # Move all layers to device.
    
    def forward(self,
                clicked_news: dict,
                candidate_news: dict):
        """
        Tensors:
            title             : (batch_size, num_news, ctx_len)
            title_mask        : (batch_size, num_news, ctx_len)
            embed             : (batch_size, num_news, ctx_len, d_embed)
            clicked_news_vec  : (batch_size, num_clicked_news, d_embed)
            candidate_news_vec: (batch_size, num_candidate_news, d_embed)
        """
        # Clicked news
        embed = self.dropout(self.embedding(clicked_news['title']['input_ids'].to(self.device)))
        clicked_news_vec = self.news_encoder(embed, clicked_news['title']['attention_mask'].to(self.device))
        final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
        # Candidate news
        embed = self.embedding(candidate_news['title']['input_ids'].to(self.device))
        candidate_news_vec = self.news_encoder(embed, candidate_news['title']['attention_mask'].to(self.device))
        # Dot product
        click_probability = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        return click_probability

# class NRMS_Glove(nn.Module):
#     """
#     NewsEncoder: (candidate_news) -> news_vector
#     UserEncoder: (clicked_news) -> user_vector
#     model      : (clicked_news, candidate_news) -> click_probability
#     """
#     def __init__(self, args: Arguments, pretrained_embedding=None) -> None:
#         super().__init__()
#         self.args = args
#         self.device = args.device
#         # ---- Layers ---- #
#         if pretrained_embedding is None:
#             self.embedding = nn.Embedding(args.vocab_size,
#                                           args.embedding_dim,
#                                           padding_idx=0) # TODO padding_idx
#         else:
#             self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
#                                                           freeze=False,
#                                                           padding_idx=0)
#         self.news_encoder = Encoder(args.num_heads, args.embedding_dim)
#         self.user_encoder = Encoder(args.num_heads, args.embedding_dim)
#         self.dropout = nn.Dropout(args.dropout_rate)
#         self.to(self.device)
    
#     def forward(self,
#                 clicked_news: dict,
#                 candidate_news: dict):
#         """
#         Args:
#             title             : (batch_size, num_news, ctx_len)
#             title_mask        : (batch_size, num_news, ctx_len)
#             embed             : (batch_size, num_news, ctx_len, d_embed)
#             clicked_news_vec  : (batch_size, num_clicked_news, d_embed)
#             candidate_news_vec: (batch_size, num_candidate_news, d_embed)
#         """
#         # Clicked news
#         embed = self.dropout(self.embedding(clicked_news['title'].to(self.device)))
#         clicked_news_vec = self.news_encoder(embed, clicked_news['title_mask'].to(self.device))
#         final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
#         # Candidate news
#         embed = self.embedding(candidate_news['title'].to(self.device))
#         candidate_news_vec = self.news_encoder(embed, candidate_news['title_mask'].to(self.device))
#         # Dot product
#         click_probability = (candidate_news_vec @ final_representation).squeeze(dim=-1)
#         return click_probability
# class NRMS_BERT(nn.Module):

#     pass

if __name__ == '__main__':
    pass