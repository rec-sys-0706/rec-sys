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

def parse_behaviors(src_path, dest_path):
    """
    Parse behaviors.tsv file into (user_id, clicked_news, candidate_news, clicked) form.
    """
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
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors['impressions'].map(lambda x: (
            ' '.join([e.split('-')[0] for e in x]),
            ' '.join([e.split('-')[1] for e in x])
        )).tolist() # tuple to list, so it can be converted to DataFrame.
    )
    # TODO
    # ! 以 behaviors[behaviors['user_id'] == 'U91836'] 為例，其中 N17059-1 有發生 click/not click 都有的現象，應該要改sample 3 個
    # print(behaviors[behaviors['user_id'] == 'U91836'].head())
    # print(behaviors[behaviors['user_id'] == 'U91836']['impressions'].iloc[0])
    behaviors.to_csv(dest_path,
                     sep='\t',
                     index=False,
                     columns=['user_id', 'clicked_news', 'candidate_news', 'clicked'])

def parse_news(config: BaseConfig,
               src_path,
               dest_dir):
    """
    Returns:
        category2int
        word2int
        news_parsed.tsv: (category, subcategory, title, abstract, title_attention_mask, abstract_attention_mask)
    """
    # TODO tqdm
    # TODO entity
    # TODO train/test mode?
    # TODO threshold
    # TODO negative sampling candidate_news




    news = pd.read_csv(src_path,
                    sep='\t',
                    names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                    index_col='news_id')
    # missing values
    news['title_entities'] = news['title_entities'].fillna('[]')
    news['abstract_entities'] = news['abstract_entities'].fillna('[]')
    news['abstract'] = news['abstract'].fillna('')

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
    for idx, (key, value) in enumerate(tf, start=2):
        # tf_threshold = 100 # TODO
        # filtered_tf = {key: value for key, value in tf.items() if value >= tf_threshold}
        word2int[key] = idx
    # ! processing news
    news = news.drop(columns=['url', 'title_entities', 'abstract_entities'])
    news['category'] = news['category'].apply(lambda c: category2int.get(c, 0))
    news['subcategory'] = news['subcategory'].apply(lambda c: category2int.get(c, 0))
    news['title'] = news['title'].apply(lambda text: [word2int.get(token, 0) for token in tokenize(text)])
    news['abstract'] = news['abstract'].apply(lambda text: [word2int.get(token, 0) for token in tokenize(text)])
    # ! truncation and padding
    news[['title', 'title_attention_mask']] = news['title'].apply(lambda t: pd.Series(tru_pad(t, config.num_tokens_title)))
    news[['abstract', 'abstract_attention_mask']] = news['abstract'].apply(lambda t: pd.Series(tru_pad(t, config.num_tokens_abstract)))
    # save to tsv
    temp = pd.DataFrame(category2int.items(), columns=['category', 'int'])
    temp.to_csv(dest_dir + '/category2int.tsv',
                sep='\t',
                index=False)
    temp = pd.DataFrame(word2int.items(), columns=['word', 'int'])
    temp.to_csv(dest_dir + '/word2int.tsv',
                sep='\t',
                index=False)
    news.to_csv(dest_dir + '/news_parsed.tsv',
                sep='\t')

if __name__ == '__main__':
    # parse_behaviors('data/MINDsmall_train/behaviors.tsv', 'data/MINDsmall_train/behaviors_parsed.tsv')
    parse_news(BaseConfig(), 'data/MINDsmall_train/news.tsv', 'data/MINDsmall_train')