import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel
from .encoder import Encoder

from parameters import Arguments

class NRMS(nn.Module):
    """
    NewsEncoder: (candidate_news) -> news_vector
    UserEncoder: (clicked_news) -> user_vector
    model      : (clicked_news, candidate_news) -> click_probability
    """
    def __init__(self, args: Arguments, vocab_size: int, pretrained_embedding=None) -> None:
        super().__init__()
        self.args = args
        self.device = args.device
        # ---- Layers ---- #
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(vocab_size,
                                          args.embedding_dim,
                                          padding_idx=0) # TODO padding_idx?
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=False,
                                                          padding_idx=0)
        self.news_encoder = Encoder(args.num_heads, args.embedding_dim)
        self.user_encoder = Encoder(args.num_heads, args.embedding_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.to(self.device) # Move all layers to device.

    def forward(self,
                clicked_news: dict,
                candidate_news: dict,
                clicked):
        """
        Tensors:
            tensors in title    : (batch_size, num_news, ctx_len)
            embed               : (batch_size, num_news, ctx_len, d_embed)
            clicked_news_vec    : (batch_size, num_clicked_news, d_embed)
            candidate_news_vec  : (batch_size, num_candidate_news, d_embed)
            final_representation: (batch_size, d_embed, 1)
            click_probability   : (batch_size, 2)
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
        return {
            'loss': F.cross_entropy(click_probability, clicked.to(self.device)),
            'logits': click_probability
        }
import pdb
class NRMS_BERT(nn.Module):
    def __init__(self, args: Arguments, pretrained_model_name):
        super().__init__()
        self.args = args
        self.device = args.device
        # ---- Layers ---- #
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.news_encoder = Encoder(args.num_heads, 768)
        self.user_encoder = Encoder(args.num_heads, 768)
        self.to(self.device) # Move all layers to device.
    def forward(self,
                clicked_news: dict,
                candidate_news: dict,
                clicked):
        
        output_dict = {key: list(torch.unbind(tensor, dim=1)) for key, tensor in clicked_news['title'].items()}
        temp = []
        for title in zip(*output_dict.values()):
            x = {
               'input_ids': title[0],
               'token_type_ids': title[1],
               'attention_mask': title[2]
            }
            bert_output = self.bert(**x)
            temp.append(bert_output.last_hidden_state)
        pdb.set_trace()
        # Clicked news
        bert_output = self.bert(**clicked_news['title'])
        pdb.set_trace() # TODO device
        embed = bert_output.last_hidden_state
        clicked_news_vec = self.news_encoder(embed, clicked_news['title']['attention_mask'].to(self.device))
        final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
        # Candidate news
        bert_output = self.bert(candidate_news['title']['input_ids'].to(self.device))
        embed = bert_output.last_hidden_state
        candidate_news_vec = self.news_encoder(embed, candidate_news['title']['attention_mask'].to(self.device))
        # Dot product
        click_probability = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        return {
            'loss': F.cross_entropy(click_probability, clicked.to(self.device)),
            'logits': click_probability
        }
if __name__ == '__main__':
    pass