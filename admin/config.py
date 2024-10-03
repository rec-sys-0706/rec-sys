import pandas as pd


news = pd.read_csv('./admin/data/news.tsv',
                    sep='\t',
                    names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                    index_col='news_id')

result = pd.read_csv('./admin/data/result.csv', index_col='user_id')