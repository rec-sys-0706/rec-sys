from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from bertviz import head_view

from transformers import AutoModel
from .encoder import Encoder
from dataset import CustomTokenizer
from parameters import Arguments

class NRMS(nn.Module):
    """
    NewsEncoder: (candidate_news) -> news_vector
    UserEncoder: (clicked_news) -> user_vector
    model      : (clicked_news, candidate_news) -> click_probability
    """
    def __init__(self,
                 args: Arguments,
                 tokenizer: CustomTokenizer,
                 next_ckpt_dir: str,
                 pretrained_embedding=None) -> None:
        super().__init__()
        self.args = args
        self.device = args.device
        # ---- Layers ---- #
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(tokenizer.vocab_size,
                                          args.embedding_dim,
                                          padding_idx=0) # TODO padding_idx?
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=False,
                                                          padding_idx=0)
        if args.use_category:
            self.category_embedding = nn.Embedding(tokenizer.num_category,
                                                   args.embedding_dim)
        self.news_encoder = Encoder(args.num_heads, args.embedding_dim)
        self.user_encoder = Encoder(args.num_heads, args.embedding_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.to(self.device) # Move all layers to device.
        self.record_vector = {
            'news_id': [],
            'vec': [],
            'category': []
        }

    def forward(self,
                user: dict,
                clicked_news: dict,
                candidate_news: dict,
                clicked=None):
        """
        Tensors:
            tensors in title    : (batch_size, num_news, ctx_len)
            embed               : (batch_size, num_news, ctx_len, d_embed)
            clicked_news_vec    : (batch_size, num_clicked_news, d_embed)
            candidate_news_vec  : (batch_size, num_candidate_news, d_embed)
            final_representation: (batch_size, d_embed, 1)
            click_probability   : (batch_size, 2)
        """
        # Category
        if self.args.use_category:
            clicked_category_embed = self.category_embedding(clicked_news['category'].to(self.device))
            candidate_category_embed = self.category_embedding(candidate_news['category'].to(self.device))
        else:
            clicked_category_embed = None
            candidate_category_embed = None
        # Clicked news
        embed = self.dropout(self.embedding(clicked_news['title']['input_ids'].to(self.device)))
        clicked_news_vec = self.news_encoder(embed, clicked_news['title']['attention_mask'].to(self.device), clicked_category_embed)
        final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
        # Candidate news
        embed = self.embedding(candidate_news['title']['input_ids'].to(self.device))
        candidate_news_vec = self.news_encoder(embed, candidate_news['title']['attention_mask'].to(self.device), candidate_category_embed)
        # Dot product
        scores = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        click_probability = F.sigmoid(scores)
        if clicked is not None:
            loss = F.binary_cross_entropy(click_probability, clicked.to(self.device))
        else:
            loss = None
        output = {
            'loss': loss,
            'logits': click_probability,
            'user': user
        }
        if self.args.mode == 'train':
            output.pop('user')
        if self.args.mode == 'valid' and self.args.generate_tsne:
            news_ids = user['clicked_news_ids']
            for i in range(len(news_ids)):
                size = min(len(news_ids[i]), len(clicked_news_vec[i]))
                self.record_vector['news_id'] += news_ids[i][:size]
                self.record_vector['vec'] += clicked_news_vec[i][:size].tolist()
                self.record_vector['category'] += user['clicked_news_category'][i][:size].tolist()
        return output
        # click_probability = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        # return {
        #     'loss': F.cross_entropy(click_probability, clicked.to(self.device)),
        #     'logits': click_probability
        # }

class NRMS_BERT(nn.Module):
    def __init__(self,
                 args: Arguments,
                 pretrained_model_name,
                 tokenizer: CustomTokenizer,
                 next_ckpt_dir: str):
        super().__init__()
        self.args = args
        self.device = args.device
        self.tokenizer = tokenizer
        # ---- Layers ---- #
        self.bert = AutoModel.from_pretrained(pretrained_model_name, output_attentions=args.generate_bertviz)
        self.news_encoder = Encoder(args.num_heads, 768)
        self.user_encoder = Encoder(args.num_heads, 768)
        if args.use_category:
            self.category_embedding = nn.Embedding(tokenizer.num_category,
                                                   args.embedding_dim)
        self.to(self.device) # Move all layers to device.
        # ---- bertviz ---- #
        self.bertviz_path = Path(next_ckpt_dir) / 'bertviz'
        if self.args.generate_bertviz:
            if not self.bertviz_path.exists():
                self.bertviz_path.mkdir()
        self.record_vector = {
            'news_id': [],
            'vec': [],
            'category': []
        }
    def forward(self,
                user: dict,
                clicked_news: dict,
                candidate_news: dict,
                clicked=None):
        # Category
        if self.args.use_category:
            clicked_category_embed = self.category_embedding(clicked_news['category'].to(self.device))
            candidate_category_embed = self.category_embedding(candidate_news['category'].to(self.device))
        else:
            clicked_category_embed = None
            candidate_category_embed = None
        # Clicked news
        batch_size, num_articles, seq_len = clicked_news['title']['input_ids'].size()
        input_ids = clicked_news['title']['input_ids'].view(-1, seq_len).to(self.device)
        attention_mask = clicked_news['title']['attention_mask'].view(-1, seq_len).to(self.device)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state.view(batch_size, num_articles, seq_len, -1)
        embed = F.dropout(last_hidden_state, p=self.args.dropout_rate, training=self.training)
        clicked_news_vec = self.news_encoder(embed, clicked_news['title']['attention_mask'].to(self.device), clicked_category_embed)
        final_representation = self.user_encoder(clicked_news_vec).unsqueeze(dim=-1)
        # Candidate news
        batch_size, num_articles, seq_len = candidate_news['title']['input_ids'].size()
        input_ids = candidate_news['title']['input_ids'].view(-1, seq_len).to(self.device)
        attention_mask = candidate_news['title']['attention_mask'].view(-1, seq_len).to(self.device)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state.view(batch_size, num_articles, seq_len, -1)
        embed = F.dropout(last_hidden_state, p=self.args.dropout_rate, training=self.training)
        candidate_news_vec = self.news_encoder(embed, candidate_news['title']['attention_mask'].to(self.device), candidate_category_embed)
        # Dot product
        scores = (candidate_news_vec @ final_representation).squeeze(dim=-1)
        click_probability = F.sigmoid(scores)
        if clicked is not None:
            loss = F.binary_cross_entropy(click_probability, clicked.to(self.device))
        else:
            loss = None
        output = {
            'loss': loss,
            'logits': click_probability,
            'user': user
        }
        if self.args.mode == 'train':
            output.pop('user')

        if self.args.mode == 'valid' and self.args.generate_tsne:
            news_ids = user['clicked_news_ids']
            for i in range(len(news_ids)):
                size = min(len(news_ids[i]), len(clicked_news_vec[i]))
                self.record_vector['news_id'] += news_ids[i][:size]
                self.record_vector['vec'] += clicked_news_vec[i][:size].tolist()
                self.record_vector['category'] += user['clicked_news_category'][i][:size].tolist()
        # if self.args.valid_test:
        #     attentions = bert_output.attentions # tuple with `num_heads` of (batch_size, num_heads, seq, seq) 
        #     stacked_tensor = torch.stack(attentions)  # Result shape: (n, m, x, y, z)
        #     list_of_tensors = list(stacked_tensor.permute(1, 0, 2, 3, 4))  # Result is a list of n tensors, each with shape (m, n, x, y, z)
        #     for ids, attn in zip(input_ids, list_of_tensors):
        #         attn = tuple(attn.unsqueeze(1))
        #         tokens = self.tokenizer.convert_ids_to_tokens(ids) 
        #         html_head_view = head_view(attn, tokens, html_action='return')
        #         try:
        #             bertviz_filename = re.sub(r'[\\/:*?"<>|]', '', " ".join(tokens))
        #             with open(self.bertviz_path / f'{bertviz_filename}.html', 'w') as file:
        #                 file.write(html_head_view.data)
        #         except:
        #             raise ValueError("Bertviz error.")
        return output
if __name__ == '__main__':
    pass