import time
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import Literal

from parameters import Arguments
from utils import CustomTokenizer, time_since, get_src_dir, get_suffix

def parse_behaviors(src_dir: Path) -> pd.DataFrame:
    """Parses `behaviors.tsv` file, generate `behaviors_parsed.csv`.
    
    Output File Format:
        The resulting CSV file will contain the following columns:
        - user_id             (str)
        - clickede_news       (list[str])
        - clicked_candidate   (list[str]): clicked candidate news set
        - unclicked_candidate (list[str]): unclicked candidate news set
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
    return behaviors

def parse_behaviors_for_test(src_dir: Path) -> pd.DataFrame:
    """Parses `behaviors.tsv` file, generate `behaviors_parsed.csv`.
    
    Output File Format:
        The resulting CSV file will contain the following columns:
        - user_id             (str)
        - clickede_news       (list[str])
        - candidate_news      (list[str])
    """
    raise ValueError("")
    behaviors = pd.read_csv(src_dir / 'behaviors.tsv',
                            sep='\t',
                            names=['impression_id', 'user_id', 'time', 'clicked_news', 'candidate_news'],
                            dtype='string',
                            index_col='impression_id')
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna('') # Handle missing values
    # behaviors['impressions'] = behaviors['impressions'].str.split() # Convert 'impressions' to list
    # behaviors = behaviors.drop(columns='time')
    # behaviors = behaviors.groupby('user_id').agg({
    #     'clicked_news': ' '.join,
    #     'impressions': lambda x: sum(x, []) # [['a', 'b'], ['c'], ['d', 'e']] -> ['a', 'b', 'c', 'd', 'e']
    # }).reset_index()
    # behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda h: list(set(h.split()))) # Remove duplicated values in 'clicked_news'
    # behaviors['impressions'] = behaviors['impressions'].apply(lambda impression: list(set(impression)))
    behaviors.to_csv(src_dir / 'behaviors_parsed.csv',
                     index=False,
                     columns=['user_id', 'clicked_news', 'candidate_news'])

def parse_news(src_dir: Path, tokenizer: CustomTokenizer) -> pd.DataFrame:
    """Parse `news.tsv`` file, generate `news_parsed.csv`.

    Output File Format:
        The resulting CSV file will contain the following columns:
        - news_id  (str)
        - category (int)
        - title    (dict[str, int])
        - abstract (dict[str, int])
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
    news['category'] = news['category'].apply(lambda c: tokenizer.encode_category(c))
    # news['subcategory'] = news['subcategory'].apply(lambda c: category2int.get(c, 0)) # TODO use subcategory?
    news['title'] = news['title'].apply(lambda text: tokenizer.encode_title(text))
    news['abstract'] = news['abstract'].apply(lambda text: tokenizer.encode_abstract(text))
    return news

def generate_word_embedding(args: Arguments, tokenizer: CustomTokenizer):
    start_time = time.time()
    print('Initializing processing of pretrained embeddings...')
    # Vocabulary
    word2int = pd.DataFrame.from_dict(tokenizer.get_vocab(), orient='index', columns=['int'])
    word2int.index.name = 'word'
    # GloVe
    print('Loading GloVe embeddings...')
    glove_embedding = pd.read_csv(args.glove_embedding_path,
                                  sep=' ',
                                  index_col=0,
                                  quoting=3,
                                  header=None,
                                  na_filter=False)
    glove_embedding.index.rename('word', inplace=True)
    if glove_embedding.shape[1] != args.embedding_dim:
        raise ValueError((
            f"Pretrained embedding source dim {glove_embedding.shape[1]} "
            f"is not equal to config.embedding_dim {args.embedding_dim}\n"
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
    missing_rows.iloc[:, 1:] = np.random.normal(size=(missing_rows.shape[0], args.embedding_dim))

    merged = word2int.merge(glove_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    result = pd.concat([merged, missing_rows]).sort_values(by='int')
    result.set_index('int', inplace=True)

    torch.save(torch.tensor(result.values, dtype=torch.float32), Path(args.train_dir) / 'pretrained_embedding.pt')
    print((
        f'Vocabulary Size  : {len(word2int)}\n'
        f'Missed Embeddings: {len(missing_rows)}\n'
        f'Intersection     : {len(merged)}\n'
        f'Missing Rate     : {len(missing_rows) / len(word2int):.4f}\n'
        f'Elapsed Time     : {time_since(start_time, "seconds"):.2f} seconds\n'
        f'Embedding file has been successfully.'
    ))

def data_preprocessing(args: Arguments, mode: Literal['train', 'valid', 'test']):
    src_dir = get_src_dir(args, mode)
    suffix = get_suffix(args)
    news_path = src_dir / f'news_parsed{suffix}.csv'
    behaviors_path = src_dir / f'behaviors_parsed{suffix}.csv'
    # Behaviors
    if not behaviors_path.exists() and args.reprocess:
        if mode == 'test':
            start = time.time()
            parse_behaviors_for_test(src_dir) # Special handling test data.
            logging.info(f"[{mode}] Parsing `behaviors.tsv` completed in {time_since(start, 'seconds'):.2f} seconds")
        elif mode in ['train', 'valid']:
            start = time.time()
            behaviors = parse_behaviors(src_dir)
            behaviors.to_csv(behaviors_path,
                        index=False,
                        columns=['user_id', 'clicked_news', 'clicked_candidate', 'unclicked_candidate'])
            logging.info(f"[{mode}] Parsing `behaviors.tsv` completed in {time_since(start, 'seconds'):.2f} seconds")
    else:
        logging.info(f"{behaviors_path} already exists.")
    # News
    if not news_path.exists() and args.reprocess:
        start = time.time()
        tokenizer = CustomTokenizer(args)
        news = parse_news(src_dir, tokenizer)
        news.to_csv(news_path)
        logging.info(f"[{mode}] Parsing `news.tsv` completed in {time_since(start, 'seconds'):.2f} seconds")
    else:
        logging.info(f"{news_path} already exists.")
    # Glove
    if mode == 'train' and args.model_name == 'NRMS-Glove':
        generate_word_embedding(args, tokenizer)

if __name__ == '__main__':
    pass