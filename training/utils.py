import string
import time
import random
from datetime import datetime
from typing import Literal

import numpy as np
import torch
import pandas as pd
import tiktoken
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from parameters import Arguments
from pydantic import BaseModel

class Encoding(BaseModel):
    input_ids: list[int]
    token_type_ids: list[int]
    attention_mask: list[int]

class GroupedNews(BaseModel):
    title: list[Encoding]
    abstract: list[Encoding]

class Example(BaseModel):
    clicked_news: GroupedNews
    candidate_news: GroupedNews
    clicked: list[int]

class EarlyStopping:
    """EarlyStopping references tensorflow.keras.callbacks.EarlyStopping."""
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.stop_training = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            (stop_training, is_better) = (False, True)
        else:
            self.counter += 1
            (stop_training, is_better) = (False, False)
            if self.counter >= self.patience:
                (stop_training, is_better) = (True, False)
                self.stop_training = True

        return stop_training, is_better

class CustomTokenizer:
    """This is deprecated, will be replaced by Huggingface.Tokenizer."""
    def __init__(self, args: Arguments):
        self.args = args
        self.mode = args.tokenizer_mode
        self.ENC = tiktoken.get_encoding("o200k_base") # gpt-4o
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.title_padding = self.tokenize_title('')
        self.abstract_padding = self.tokenize_abstract('')

    def __call__(self, text):
        pass

    def tokenize(self, text: str) -> list[str]:
        # return [ENC.decode([token]) for token in tokens] # TODO optmize
        # return list(text.lower()) # character only, 123
        # return re.sub(r'[^a-z0-9\s]', '', text.lower()).split() # 36306
        # return nltk.word_tokenize(text.lower()) # 37539
        pass
        # For encoding: [int(word2int.get(token, 0)) for token in tokenize(text)]

    def encode(self, text: str) -> Encoding:
        token_ids = self.ENC.encode(text)
        return token_ids

    def encode_title(self, text) -> Encoding:
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=self.args.num_tokens_title)
    def encode_abstract(self, text) -> Encoding:
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=self.args.num_tokens_abstract)

    def decode():
        pass

    def __build_tokenizer():
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer()
        data = [""]
        pass
    # TODO build tokenizer

def time_since(base: float, format: None|Literal['seconds']=None):
    now = time.time()
    elapsed_time = now - base
    if format == 'seconds':
        return elapsed_time
    else:
        return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

def detokenize(word2int_path, seq):
    seq = np.array(seq).astype(str)
    word2int = dict(pd.read_csv(word2int_path,
                                sep='\t',
                                names=['word', 'int'],
                                index_col=False).values)
    int2word = {v: k for k, v in word2int.items()}

    decode_map = np.vectorize(int2word.get)
    return decode_map(seq)

def get_datetime_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S")

def format_duration(sec):
    return time.strftime("%H:%M:%S", time.gmtime(sec))

def fix_all_seeds(seed):
    """Fixes RNG seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def tru_pad(tokens: list[str], max_length: int):
    """truncation and padding"""
    if len(tokens) < max_length:
        result = tokens + [0] * (max_length - len(tokens))
    else:
        result = tokens[:max_length]
    attention_mask = [1 if i < len(tokens) else 0 for i in range(max_length)]
    return result, attention_mask
    # ! truncation and padding
    # news[['title', 'title_attention_mask']] = news['title'].apply(lambda t: pd.Series(tru_pad(t, args.num_tokens_title)))
    # news[['abstract', 'abstract_attention_mask']] = news['abstract'].apply(lambda t: pd.Series(tru_pad(t, args.num_tokens_abstract)))

def test_string() -> str:
    mixed_case = "No"
    accents = "HÉLLOcafé"
    unicode = "你好こんにちは😊"
    control = r"\x00\x07"
    currency = "$100 €50 ₹200"
    full_width = "Ｈｅｌｌｏ"
    url = "http://example.com"
    file_path = "C:\\Users\\Name"
    text = (f'   {mixed_case}, {accents}, {unicode}, {string.punctuation}, {control}, {currency}, {full_width}, مرحبا بالعالم , 𝒜𝓁𝓅𝒽𝒶, {url}'
            f', {file_path},½ ⅓ H₂O x² ∑ √ œuvre'
            f'<body></body>   ')
    return text


def list_to_dict(objs: list[dict]):
    """Convert list[dict] to dict[list]"""
    result = {}

    for obj in objs:
        for key, value in obj.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    return result

def dict_to_tensors(obj: dict):
    """Convert dictionary values to tensors recursively"""
    for key, value in obj.items():
        if isinstance(value, dict): # is a dictionary
            dict_to_tensors(value) # recursively
        else:
            obj[key] = torch.tensor(value)
    return obj

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items) # TODO

if __name__ == '__main__':
    pass