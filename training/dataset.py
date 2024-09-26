from ast import literal_eval
import logging
import random
import pdb
from typing import Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from parameters import Arguments
from utils import CustomTokenizer, Example, list_to_dict, dict_to_tensors, get_src_dir



class NewsDataset(Dataset):
    """Sample data form `behaviors.tsv` and create dataset based on `news.tsv`
    """
    # For each element {
    #     clicked_news       : {
    #         title          : (batch_size, num_clicked_news_a_user, num_tokens_title)
    #     }
    #     candidate_news     : {
    #         title          : (batch_size, 1 + k, num_tokens_title)
    #     }
    #     clicked(valid/test): (batch_size, 1 + k)
    # } # TODO
    def __init__(self, args: Arguments, tokenizer: CustomTokenizer, mode) -> None:
        random.seed(args.seed)
        src_dir = get_src_dir(args, mode)
        result_path = src_dir / f'{mode}.pt'
        if result_path.exists():
            data = torch.load(result_path)
            self.result = data['result']
        else:
            logging.info(f"Cannot locate file {mode}.pt in '{src_dir}'.")
            logging.info(f"Starting data processing.")
            __news = pd.read_csv(src_dir / 'news_parsed.csv', index_col='news_id')
            __news['title'] = __news['title'].apply(literal_eval)
            __news['abstract'] = __news['abstract'].apply(literal_eval) # literal_eval first, this is an improvement from 05:00 to 00:15
            __news = __news.to_dict(orient='index')
            __behaviors = pd.read_csv(src_dir / 'behaviors_parsed.csv')

            if args.drop_insufficient:
                __behaviors = __behaviors.dropna(subset=['clicked_news'])

            def get_news(news_id):
                title = tokenizer.title_padding if news_id is None else __news[news_id]['title']
                abstract = tokenizer.abstract_padding if news_id is None else __news[news_id]['abstract']
                return title, abstract
            def get_grouped_news(news_ids):
                title_list = []
                abstract_list = []
                for news_id in news_ids:
                    title, abstract = get_news(news_id)
                    title_list.append(title)
                    abstract_list.append(abstract)
                return dict_to_tensors(list_to_dict(title_list), torch.int), dict_to_tensors(list_to_dict(abstract_list), torch.int) # Reduce memory usage

            __behaviors = __behaviors.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            result = []
            for idx, row in tqdm(__behaviors.iterrows(), total=len(__behaviors)):
                if idx == args.max_dataset_size:
                    break
                clicked_news_ids = literal_eval(row['clicked_news'])
                random.shuffle(clicked_news_ids)
                clicked_news_ids = clicked_news_ids[:args.num_clicked_news] # truncate clicked_news
                num_missing_news = args.num_clicked_news - len(clicked_news_ids)

                clicked_candidate_ids = literal_eval(row['clicked_candidate'])
                unclicked_candidate_ids = literal_eval(row['unclicked_candidate'])
                if args.drop_insufficient:
                    if len(clicked_candidate_ids) < 1 or len(unclicked_candidate_ids) < args.negative_sampling_ratio:
                        continue # skip if row is not completed
                    if len(clicked_news_ids) < 1:
                        continue
                
                # clicked_news_ids += [None] * num_missing_news # padding clicked_news
                candidate_news_ids = random.sample(clicked_candidate_ids, 1) + random.sample(unclicked_candidate_ids, args.negative_sampling_ratio)

                clicked_title, clicked_abstract = get_grouped_news(clicked_news_ids)
                candidate_title, candidate_abstract = get_grouped_news(candidate_news_ids)

                clicked_news = {
                    'title': clicked_title,
                    # 'abstract': clicked_abstract
                }
                
                candidate_news = {
                    'title': candidate_title,
                    # 'abstract': candidate_abstract
                }
                result.append({
                    'user_id': row['user_id'],
                    'clicked_news': clicked_news,
                    'candidate_news': candidate_news,
                    'clicked': torch.tensor([1] + [0] * args.negative_sampling_ratio, dtype=torch.float32) # TODO if mode == test
                    # ! important for [RuntimeError: Expected floating point type for target with class probabilities, got Long]
                })
            # Save
            torch.save({
                'result': result
            }, result_path)
            logging.info(f"{mode}.pt saved successfully at {result_path}.")
            self.result = result
    def __len__(self):
        return len(self.result)
    def __getitem__(self, index):
        return self.result[index]

@dataclass
class CustomDataCollator:
    tokenizer: CustomTokenizer

    def __call__(self, batch: list[Example]) -> dict[str, Any]:
        """
        DataLoader will shuffle and sample features(batch), and input `features` into DataCollator,
        DataCollator is just a callable function, given a batch, return a processed batch.

        batch: {
            clicked_news: list[GroupedNews]
            candidate_news: list[GroupedNews]
            clicked: list[list[int]]
        }

        output: {
            clicked_news: {
                title   : Encoding with (batch_size, num_clicked_news, num_title_tokens)
                abstract: Encoding with (batch_size, num_clicked_news, num_abstract_tokens)
            }
            candidate_news: {
                title   : Encoding with (batch_size, k+1, num_title_tokens)
                abstract: Encoding with (batch_size, k+1, num_abstract_tokens)
            }
            clicked     : (batch_size, k+1)
        }
        """ # TODO
        max_num_titles = max([len(item['clicked_news']['title']['input_ids']) for item in batch])
        # TODO or maybe mean

        result = {
            'clicked_news': {
                'title': {
                    'input_ids': [],
                    'attention_mask': []
                }
            },
            'candidate_news': {
                'title': {
                    'input_ids': [],
                    'attention_mask': []
                }
            },
            'clicked': torch.stack([example['clicked'] for example in batch], dim=0)
        }

        for item in batch:
            # Clicked News
            encodes = item['clicked_news']['title'] # encodes of title of clicked_news
            num_titles = len(encodes['input_ids'])
            if num_titles < max_num_titles:
                padding_input_ids = torch.tensor([self.tokenizer.title_padding['input_ids']] * (max_num_titles - num_titles))
                padding_attention_mask = torch.tensor([self.tokenizer.title_padding['attention_mask']] * (max_num_titles - num_titles))
                # TODO, title_padding直接轉tensor
                encodes['input_ids'] = torch.cat((encodes['input_ids'], padding_input_ids), dim=0)
                encodes['attention_mask'] = torch.cat((encodes['attention_mask'], padding_attention_mask), dim=0)
            result['clicked_news']['title']['input_ids'].append(encodes['input_ids'])
            result['clicked_news']['title']['attention_mask'].append(encodes['attention_mask'])
            # Candidate News
            encodes = item['candidate_news']['title'] # encodes of title
            result['candidate_news']['title']['input_ids'].append(encodes['input_ids'])
            result['candidate_news']['title']['attention_mask'].append(encodes['attention_mask'])
        result['clicked_news']['title']['input_ids'] = torch.stack(result['clicked_news']['title']['input_ids'])
        result['clicked_news']['title']['attention_mask'] = torch.stack(result['clicked_news']['title']['attention_mask'])
        result['candidate_news']['title']['input_ids'] = torch.stack(result['candidate_news']['title']['input_ids'])
        result['candidate_news']['title']['attention_mask'] = torch.stack(result['candidate_news']['title']['attention_mask'])
        return result

if __name__ == '__main__':
    pass