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
from config import BaseConfig
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time
import torch
import time
from utils import time_since

def tokenize(text: str) -> list[str]:
    # return list(set(text)) # character only, 123
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).split() # 36306
    # return nltk.word_tokenize(text.lower()) # 37539

def tru_pad(tokens: list[str], max_length: int):
    """truncation and padding"""
    if len(tokens) < max_length:
        result = tokens + [0] * (max_length - len(tokens))
    else:
        result = tokens[:max_length]
    attention_mask = [1 if i < len(tokens) else 0 for i in range(max_length)]
    return result, attention_mask

def sample(true_set: set[str], false_set: set[str], negative_sampling_ratio=4) -> tuple[str, str]:
    """negative sampling"""
    true_set = list(true_set)
    false_set = list(false_set)

    random.shuffle(true_set)
    random.shuffle(false_set)

    true_set = true_set[:1]
    false_set = false_set[:negative_sampling_ratio]

    # assert len(true_set) == 1 and len(false_set) == negative_sampling_ratio
    news_id = ' '.join(true_set + false_set)
    clicked = ' '.join('1' * len(true_set) + '0' * len(false_set))
    return (news_id, clicked)

def parse_behaviors(config: BaseConfig, mode):
    """
    Parse behaviors.tsv file into (user_id, clicked_news, candidate_news, clicked) form.
    """
    if mode == 'train':
        src_dir = Path(config.train_dir)
    elif mode == 'valid':
        src_dir = Path(config.val_dir)
    elif mode == 'test':
        src_dir = Path(config.test_dir)
    else:
        raise ValueError(f"[ERROR] Expected 'mode' be str['train'|'valid'|'test'] but got '{mode}'.")

    src_path = src_dir / 'behaviors.tsv'
    if not src_path.exists():
        raise FileNotFoundError(f"Cannot locate behaviors.tsv file in '{src_dir}'")
    # Read file
    behaviors = pd.read_csv(src_path,
                            sep='\t',
                            names=['impression_id', 'user_id', 'time', 'clicked_news', 'impressions'],
                            dtype='string',
                            index_col='impression_id')
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna('') # Handle missing values.
    behaviors['impressions'] = behaviors['impressions'].str.split() # Convert 'impressions' to list.
    behaviors = behaviors.drop(columns='time')
    behaviors = behaviors.groupby('user_id').agg({
        'clicked_news': ' '.join,
        'impressions': lambda x: sum(x, []) # [['a', 'b'], ['c'], ['d', 'e']] -> ['a', 'b', 'c', 'd', 'e']
    }).reset_index()
    behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda h: ' '.join(set(h.split()))) # Remove duplicated values in 'clicked_news'.
    behaviors['impressions'] = behaviors['impressions'].apply(lambda impression: list(set(impression)))

    # Create candidate_news & clicked columns
    k = config.negative_sampling_ratio
    behaviors[['candidate_news', 'clicked']] = [None, None]
    for idx, row in tqdm(behaviors['impressions'].items(), total=len(behaviors)):
        true_set = set()
        false_set = set()
        for e in row:
            news_id, clicked = e.split('-') # TODO assert size 2
            true_set.add(news_id) if clicked == '1' else false_set.add(news_id)
        false_set -= true_set # duplicated news_id keep by true_set
        behaviors.loc[idx, ['candidate_news', 'clicked']] = sample(true_set, false_set, k) # TODO config

    
    behaviors = behaviors[behaviors['clicked'].apply(len) == 2*k+1]
    # behaviors['clicked'].apply(len).sort_values() # TODO delete
    # [~behaviors['clicked'].str.contains('1')] # 不包含 1 # TODO delete

    # print(behaviors[behaviors['user_id'] == 'U91836'].head()) # TODO delete
    # print(behaviors[behaviors['user_id'] == 'U91836']['impressions'].iloc[0]) # TODO delete
    behaviors.to_csv(src_dir / 'behaviors_parsed.tsv',
                     sep='\t',
                     index=False,
                     columns=['user_id', 'clicked_news', 'candidate_news', 'clicked'])

def parse_news(config: BaseConfig,
               mode: str):
    """Parse news

    if mode=='train', it will save category2int.tsv, word2int.tsv to test_dir.
    else model=='test', it load category2int.tsv, word2int.tsv from train_dir.

    Args:
        config
        src_path: source news filepath.
        dest_dir: destination directory to save/load category2int.tsv, word2int.tsv.
        mode: as description.

    Returns:
        category2int
        word2int
        news_parsed.tsv: (category, subcategory, title, abstract, title_attention_mask, abstract_attention_mask)
    """
    # TODO entity

    if mode == 'train':
        src_dir = Path(config.train_dir)
    elif mode == 'valid':
        src_dir = Path(config.val_dir)
    elif mode == 'test':
        src_dir = Path(config.test_dir)
    else:
        raise ValueError(f"[ERROR] Expected 'mode' be str['train'|'valid'|'test'] but got '{mode}'.")
    src_path = src_dir / 'news.tsv'
    if not src_path.exists():
        raise FileNotFoundError(f"Cannot locate news.tsv file in '{src_dir}'")

    # Read file
    news = pd.read_csv(src_path,
                    sep='\t',
                    names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                    index_col='news_id')
    # missing values
    news['title_entities'] = news['title_entities'].fillna('[]')
    news['abstract_entities'] = news['abstract_entities'].fillna('[]')
    news['abstract'] = news['abstract'].fillna('')

    if mode == 'train': # Create category2int and word2int, then save.
        # ! processing category
        categories = pd.concat([news['category'], news['subcategory']]).unique()
        category2int = {
            "<pad>": 0
        }
        category2int.update({category: idx for idx, category in enumerate(categories, 1)})
        # ! processing words
        tokens = []
        for text in pd.concat([news['title'], news['abstract']]):
            tokens += tokenize(text) # list concat
            # TODO [optimize]
            # token = df[text_column].apply(tokenize)
            # tokens = sum(token.tolist(), [])
        tf = Counter(tokens).most_common() # term frequency
        word2int = {
            "<pad>": 0,
            "<unk>": 1,
        }
        for idx, (key, count) in enumerate(tf, start=2):
            if count < config.tf_threshold:
                break
            word2int[key] = idx
    elif mode in {'valid', 'test'}: # Load category2int and word2int.
        category2int_path = Path(config.train_dir) / 'category2int.tsv'
        word2int_path = Path(config.train_dir) / 'word2int.tsv'
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
    # ! processing news
    news = news.drop(columns=['url', 'title_entities', 'abstract_entities'])
    news['category'] = news['category'].apply(lambda c: category2int.get(c, 0))
    news['subcategory'] = news['subcategory'].apply(lambda c: category2int.get(c, 0))
    news['title'] = news['title'].apply(lambda text: [int(word2int.get(token, 0)) for token in tokenize(text)])
    news['abstract'] = news['abstract'].apply(lambda text: [int(word2int.get(token, 0)) for token in tokenize(text)])
    # ! truncation and padding
    news[['title', 'title_attention_mask']] = news['title'].apply(lambda t: pd.Series(tru_pad(t, config.num_tokens_title)))
    news[['abstract', 'abstract_attention_mask']] = news['abstract'].apply(lambda t: pd.Series(tru_pad(t, config.num_tokens_abstract)))
    # save to tsv
    if mode == 'train':
        (pd.DataFrame(category2int.items(), columns=['category', 'int'])
         .to_csv(src_dir / 'category2int.tsv',
                 sep='\t',
                 index=False))
        (pd.DataFrame(word2int.items(), columns=['word', 'int'])
         .to_csv(src_dir / 'word2int.tsv',
                 sep='\t',
                 index=False))
    news.to_csv(src_dir / 'news_parsed.tsv',
                sep='\t')

def generate_word_embedding(config: BaseConfig):
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
if __name__ == '__main__':
    # parse_behaviors(BaseConfig(), 'valid')
    parse_news(BaseConfig(), 'train')
    # generate_word_embedding(BaseConfig())