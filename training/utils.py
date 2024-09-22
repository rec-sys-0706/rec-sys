import string
import time
import random
from datetime import datetime
from typing import Literal
from pathlib import Path
import logging

import numpy as np
import torch
import pandas as pd
import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BertTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
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
        self.SPECIAL_TOKENS = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }
        self.args = args
        self.mode = args.tokenizer_mode
        self.tokenizer_file = Path(args.train_dir) / 'tokenizer.json'
        self.categorizer_file = Path(args.train_dir) / 'categorizer.json'

        self.__categorizer = self.__build_categorizer()
        if self.mode == 'vanilla':
            if not self.tokenizer_file.exists():
                logging.info("Tokenizer file not detected.")
                logging.info("Start building tokenizer and saving to `train_dir`.")
                self.__tokenizer = self.__build_tokenizer()
            else:
                self.__tokenizer = self.__load_tokenizer()
            self.__tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        elif self.mode == 'bert':
            self.__tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            pass
            # self.ENC = tiktoken.get_encoding("o200k_base") # gpt-4o

        self.title_padding = self.encode_title('')
        self.abstract_padding = self.encode_abstract('')
        self.vocab_size = self.__tokenizer.vocab_size

    def __call__(self, *args, **kwargs):
        return self.__tokenizer(*args, **kwargs)

    def tokenize(self, text: str) -> list[str]:
        # return [ENC.decode([token]) for token in tokens] # TODO optmize
        # return list(text.lower()) # character only, 123
        # return re.sub(r'[^a-z0-9\s]', '', text.lower()).split() # 36306
        # return nltk.word_tokenize(text.lower()) # 37539
        pass
        # For encoding: [int(word2int.get(token, 0)) for token in tokenize(text)]

    def encode(self):
        pass
    def encode_title(self, text) -> Encoding:
        return self.__tokenizer(text, padding='max_length', truncation=True, max_length=self.args.num_tokens_title)
    def encode_abstract(self, text) -> Encoding:
        return self.__tokenizer(text, padding='max_length', truncation=True, max_length=self.args.num_tokens_abstract)
    def encode_category(self, category):
        return self.__categorizer.vocab.get(category, 0)
    def decode():
        pass

    def __build_tokenizer(self) -> PreTrainedTokenizerFast:
        args = self.args
        news = pd.read_csv(Path(args.train_dir) / 'news.tsv',
                        sep='\t',
                        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                        index_col='news_id')
        news['abstract'] = news['abstract'].fillna('')
        texts = pd.concat((news['title'], news['abstract'])).tolist()

        tokenizer = Tokenizer(models.WordLevel(unk_token=self.SPECIAL_TOKENS['unk_token']))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Lowercase(), normalizers.StripAccents()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=args.max_vocab_size,
                                            min_frequency=args.min_frequency,
                                            special_tokens=list(self.SPECIAL_TOKENS.values()))

        tokenizer.train_from_iterator(texts, trainer)
        tokenizer.save(self.tokenizer_file.__str__())
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        return tokenizer

    def __load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file.__str__())
        return tokenizer

    def __build_categorizer(self) -> PreTrainedTokenizerFast:
        args = self.args
        news = pd.read_csv(Path(args.train_dir) / 'news.tsv',
                        sep='\t',
                        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                        index_col='news_id')

        categories = pd.concat([news['category'], news['subcategory']]).unique().tolist()
        vocab = {category: idx for idx, category in enumerate(categories, start=1)}
        vocab.update({'<unk>': 0})
        _categorizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        _categorizer.pre_tokenizer = pre_tokenizers.Whitespace()
        _categorizer.save(self.categorizer_file.__str__())

        categorizer = PreTrainedTokenizerFast(tokenizer_object=_categorizer, unk_token="<unk>")
        return categorizer
    def save_pretrained(self, *args, **kwargs):
        self.__tokenizer.save_pretrained(*args, **kwargs)

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

def get_src_dir(args: Arguments, mode) -> Path:
    if mode == 'train':
        src_dir = Path(args.train_dir)
    elif mode == 'valid':
        src_dir = Path(args.val_dir)
    elif mode == 'test':
        src_dir = Path(args.test_dir)
    else:
        raise ValueError(f"[ERROR] Expected `mode` be str['train'|'valid'|'test'] but got `{mode}` instead.")
    return src_dir

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

def dict_to_tensors(obj: dict, dtype=None):
    """Convert dictionary values to tensors recursively"""
    for key, value in obj.items():
        if isinstance(value, dict): # is a dictionary
            dict_to_tensors(value, dtype) # recursively
        else:
            obj[key] = torch.tensor(value, dtype=dtype)
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
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    from parameters import parse_args
    args = parse_args()
    tokenizer = CustomTokenizer(args)


    fake_texts = [
        "Hello, how are you?",
        "This is a test sentence for demonstrating the data collator.",
        "Short sentence.",
        "我好愛你哦妳這個笨蛋呆頭，橄乾羭茰"
    ]
    print(tokenizer(fake_texts, padding=True))
