"""
Returns:
    Generate processed files including
    1. behaviors_parsed.tsv
    2. news_parsed.tsv
    3. word2int.tsv
    4. category2int.tsv
"""
import pandas as pd
import re
from collections import Counter
from parameters import parse_args, Arguments
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time
import torch
import time
from utils import time_since, CustomTokenizer
from typing import Literal


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

def parse_behaviors(src_dir: Path):
    """Parses behaviors.tsv file
    
    Output File Format:
        The resulting CSV file will contain the following columns:
        - `user_id`             (str)
        - `clickede_news`       (list[News])
        - `clicked_candidate`   (list[News]): clicked candidate news
        - `unclicked_candidate` (list[News]): unclicked candidate news
    """

    behaviors = pd.read_csv(src_dir / 'behaviors.tsv',
                            sep='\t',
                            names=['impression_id', 'user_id', 'time', 'clicked_news', 'impressions'],
                            dtype='string',
                            index_col='impression_id')
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna('') # Handle missing values
    behaviors['impressions'] = behaviors['impressions'].str.split() # Convert 'impressions' to list
    behaviors = behaviors.drop(columns='time')
    behaviors = behaviors.groupby('user_id').agg({
        'clicked_news': ' '.join,
        'impressions': lambda x: sum(x, []) # [['a', 'b'], ['c'], ['d', 'e']] -> ['a', 'b', 'c', 'd', 'e']
    }).reset_index()
    behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda h: list(set(h.split()))) # Remove duplicated values in 'clicked_news'
    behaviors['impressions'] = behaviors['impressions'].apply(lambda impression: list(set(impression)))

    # Create clicked & unclicked columns
    behaviors[['clicked', 'unclicked']] = [None, None]
    for idx, row in tqdm(behaviors['impressions'].items(), total=len(behaviors)):
        candidate_news = {}
        for e in row:
            news_id, clicked = e.split('-') # ! Raise error is size != 2
            if clicked == '1' or news_id not in candidate_news: # Prevent duplicated news_id
                candidate_news[news_id] = clicked
        # Separate clicked and unclicked candidate_news
        true_list = []
        false_list = []
        for news_id, clicked in candidate_news.items():
            if clicked == '1':
                true_list.append(news_id)
            elif clicked == '0':
                false_list.append(news_id)
            else:
                raise ValueError("An unexpected error has occurred at data processing phase.")
        behaviors.loc[idx, ['clicked_candidate', 'unclicked_candidate']] = str(true_list), str(false_list) # Ground true candidate_news

    behaviors.to_csv(src_dir / 'behaviors_parsed.csv',
                     index=False,
                     columns=['user_id', 'clicked_news', 'clicked_candidate', 'unclicked_candidate'])

def build_tokenizer(args: Arguments) -> CustomTokenizer:
    """
    Args:
        src_dir (Path)
        config  (BaseConfig)    : for `tf_threshold`, `num_tokens_title`, and `num_tokens_abstract`
        mode    (Literal['train', 'valid', 'test'])         

    If mode=='train', it will save category2int.tsv, word2int.tsv to `src_dir`.
    Otherwise model==['valid', 'test'], don't build.
    """
    news = pd.read_csv(Path(args.train_dir) / 'news.tsv',
                       sep='\t',
                       names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                       index_col='news_id')
    print(pd.concat((news['title'], news['abstract'])).iloc[0]) # TODO
    # # ! processing category
    # categories = pd.concat([news['category'], news['subcategory']]).unique()
    # category2int = {
    #     "<pad>": 0
    # }
    # category2int.update({category: idx for idx, category in enumerate(categories, 1)})
    # # ! processing words
    # tokens = []
    # for text in pd.concat([news['title'], news['abstract']]):
    #     tokens += tokenize(text) # list concat
    #     # TODO [optimize]
    #     # token = df[text_column].apply(tokenize)
    #     # tokens = sum(token.tolist(), [])
    # tf = Counter(tokens).most_common() # term frequency
    # word2int = {
    #     "<pad>": 0,
    #     "<unk>": 1,
    # }
    # for idx, (key, count) in enumerate(tf, start=2):
    #     if count < args.tf_threshold:
    #         break
    #     word2int[key] = idx

    return

    # Handle missing values
    news['title_entities'] = news['title_entities'].fillna('[]')
    news['abstract_entities'] = news['abstract_entities'].fillna('[]')
    news['abstract'] = news['abstract'].fillna('')
    """ When use `tokenize` funciton...
    if mode == 'train': # Create category2int and word2int, then save.

    elif mode in {'valid', 'test'}: # Load category2int and word2int from `train_dir`
        category2int_path = Path(args.train_dir) / 'category2int.tsv'
        word2int_path = Path(args.train_dir) / 'word2int.tsv'
        if not category2int_path.exists():
            raise FileNotFoundError(f"File '{category2int_path}' does not exist.")
        elif not word2int_path.exists():
            raise FileNotFoundError(f"File '{word2int_path}' does not exist.")
        category2int = dict(pd.read_csv(category2int_path,
                                        sep='\t',
                                        index_col=False).values)
        word2int = dict(pd.read_csv(word2int_path,
                                    sep='\t',
                                    index_col=False,
                                    na_filter=False).values)
    """
    # if mode == 'train':
    #     (pd.DataFrame(category2int.items(), columns=['category', 'int'])
    #      .to_csv(src_dir / 'category2int.tsv',
    #              sep='\t',
    #              index=False))
    #     (pd.DataFrame(word2int.items(), columns=['word', 'int'])
    #      .to_csv(src_dir / 'word2int.tsv',
    #              sep='\t',
    #              index=False))

# TODO use entity?
def parse_news(src_dir: Path, tokenizer: CustomTokenizer) -> tuple[dict, dict]:
    """Parse news.tsv to tokenized text.
    Output File Format:
        The resulting CSV file will contain the following columns:
        - category: int
        - title   : list[int]
        - abstract: list[int]
    """
    news = pd.read_csv(src_dir / 'news.tsv',
                       sep='\t',
                       names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                       index_col='news_id')
    # Handle missing values
    news['abstract'] = news['abstract'].fillna('') # TODO drop abstract None?

    # ! processing news
    news = news.drop(columns=['subcategory', 'url', 'title_entities', 'abstract_entities'])
    news = news.sort_index()
    # news['category'] = news['category'].apply(lambda c: category2int.get(c, 0)) # TODO category2int in tokenizer.
    # news['subcategory'] = news['subcategory'].apply(lambda c: category2int.get(c, 0))
    news['title'] = news['title'].apply(lambda text: tokenizer.tokenize_title(text))
    news['abstract'] = news['abstract'].apply(lambda text: tokenizer.tokenize_abstract(text)) # TODO Don't need to tokenize here.
    news.to_csv(src_dir / 'news_parsed.csv')

def generate_word_embedding(config: Arguments):
    start_time = time.time()
    print('Initializing processing of pretrained embeddings...')
    # Vocabulary
    word2int_path = Path(config.train_dir) / 'word2int.tsv'
    if not word2int_path.exists():
        raise FileNotFoundError(f"File '{word2int_path}' does not exist.")
    word2int = pd.read_csv(word2int_path,
                        sep='\t',
                        index_col='word',
                        na_filter=False)
    # GloVe
    print('Loading GloVe embeddings...')
    glove_embedding = pd.read_csv(config.glove_embedding_path,
                                sep=' ',
                                index_col=0,
                                quoting=3,
                                header=None,
                                na_filter=False)
    glove_embedding.index.rename('word', inplace=True)
    if glove_embedding.shape[1] != config.embedding_dim:
        raise ValueError((
            f"Pretrained embedding source dim {glove_embedding.shape[1]} "
            f"is not equal to config.embedding_dim {config.embedding_dim}\n"
            "Please check your config.py file."
        ))
    print(f'GloVe embeddings loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')
    # Missing Rows
    temp = word2int.merge(glove_embedding,
                          how='left',
                          indicator=True,
                          left_index=True,
                          right_index=True)
    missing_rows = temp[temp['_merge'] == 'left_only'].drop(columns='_merge')
    missing_rows.iloc[:, 1:] = np.random.normal(size=(missing_rows.shape[0], config.embedding_dim))

    merged = word2int.merge(glove_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    result = pd.concat([merged, missing_rows]).sort_values(by='int')
    result.set_index('int', inplace=True)

    torch.save(torch.tensor(result.values, dtype=torch.float32), Path(config.train_dir) / 'pretrained_embedding.pt')
    print((
        f'Vocabulary Size  : {len(word2int)}\n'
        f'Missed Embeddings: {len(missing_rows)}\n'
        f'Intersection     : {len(merged)}\n'
        f'Missing Rate     : {len(missing_rows) / len(word2int):.4f}\n'
        f'Elapsed Time     : {time_since(start_time, "seconds"):.2f} seconds\n'
        f'Embedding file has been successfully.'
    ))


def data_preprocessing(args: Arguments, mode: Literal['train', 'valid', 'test']):
    """Parse behaviors.tsv and news.tsv into behaviors_parsed.tsv and news_parsed.tsv""" # TODO
    start = time.time()
    src_dir = get_src_dir(args, mode)
    # parse_behaviors(src_dir)
    print(time_since(start))
    tokenizer = CustomTokenizer(args) # TODO if using glove or nltk, must build tokenizer first.
    parse_news(src_dir, tokenizer)
    print(time_since(start))

if __name__ == '__main__':
    args = parse_args()
    build_tokenizer(args)
    # data_preprocessing(args, 'train')
    # # TODO generate_word_embedding random