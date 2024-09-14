import time
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Literal
import tiktoken


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
    def __init__(self, name: Literal['gpt-4o']='gpt-4o'):
        self.ENC = tiktoken.get_encoding("o200k_base") # gpt-4o

    def tokenize(self, text: str) -> list[str]:
        # return [ENC.decode([token]) for token in tokens] # TODO optmize
        # return list(text.lower()) # character only, 123
        # return re.sub(r'[^a-z0-9\s]', '', text.lower()).split() # 36306
        # return nltk.word_tokenize(text.lower()) # 37539
        pass
        # For encoding: [int(word2int.get(token, 0)) for token in tokenize(text)]

    def encode(self, text: str) -> list[int]:
        token_ids = self.ENC.encode(text)
        return token_ids

    def decode():
        pass

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

def get_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S")

def format_duration(sec):
    return time.strftime("%H:%M:%S", time.gmtime(sec))
    
def fix_all_seeds(seed):
    '''Fixes RNG seeds for reproducibility.'''
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
if __name__ == '__main__':
    print(get_now())