from ast import literal_eval
import logging
import random
from typing import Any
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from parameters import Arguments
from utils import CustomTokenizer, list_to_dict, dict_to_tensors, get_src_dir, get_suffix


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
    def __init__(self,
                 args: Arguments,
                 tokenizer: CustomTokenizer,
                 mode: Literal['train', 'valid', 'test'],
                 use_full_candidate: bool=False) -> None:
        random.seed(args.seed)
        src_dir = get_src_dir(args, mode)
        suffix = get_suffix(args)
        news_path = src_dir / f'news_parsed{suffix}.csv'
        behaviors_path = src_dir / f'behaviors_parsed{suffix}.csv'
        
        result_path = src_dir / f'{mode}{suffix}.pt'
        if result_path.exists() and not args.regenerate_dataset:
            self.result = torch.load(result_path)
        else:
            if not result_path.exists():
                logging.info(f"Cannot locate file {mode}.pt in '{src_dir}'.")
                logging.info(f"Starting data processing.")
            elif args.regenerate_dataset:
                logging.info(f"Regenerating {mode}.pt.")
            __news = pd.read_csv(news_path, index_col='news_id')
            __news['title'] = __news['title'].apply(literal_eval)
            __news['abstract'] = __news['abstract'].apply(literal_eval) # literal_eval first, this is an improvement from 05:00 to 00:15
            __news = __news.to_dict(orient='index')
            __behaviors = pd.read_csv(behaviors_path)

            if args.drop_insufficient:
                __behaviors = __behaviors.dropna(subset=['clicked_news'])

            def get_news(news_id):
                title = tokenizer.title_padding if news_id is None else __news[news_id]['title']
                abstract = tokenizer.abstract_padding if news_id is None else __news[news_id]['abstract']
                category = 0 if news_id is None else __news[news_id]['category']
                return title, abstract, category
            def get_grouped_news(news_ids):
                title_list = []
                abstract_list = []
                category_list = []
                for news_id in news_ids:
                    title, abstract, category = get_news(news_id)
                    title_list.append(title)
                    abstract_list.append(abstract)
                    category_list.append(category)
                
                return (
                    dict_to_tensors(list_to_dict(title_list), torch.int),
                    dict_to_tensors(list_to_dict(abstract_list), torch.int), # ? Reduce memory usage?
                    torch.tensor(category_list, dtype=torch.int)
                )
            __behaviors = __behaviors.sample(frac=1, random_state=args.seed).reset_index(drop=True) # suffle
            result = []
            for idx, row in tqdm(__behaviors.iterrows(), total=len(__behaviors)):
                if idx == args.max_dataset_size:
                    break # limit dataset size
                clicked_news_ids = literal_eval(row['clicked_news'])
                random.shuffle(clicked_news_ids) # random
                clicked_news_ids = clicked_news_ids[:args.max_clicked_news] # truncate clicked_news
                num_missing_news = args.max_clicked_news - len(clicked_news_ids)

                # Generate candidate_ids
                if mode in ['train', 'valid']:
                    clicked_candidate_ids = literal_eval(row['clicked_candidate'])
                    unclicked_candidate_ids = literal_eval(row['unclicked_candidate'])
                    if args.drop_insufficient:
                        if len(clicked_candidate_ids) < 1 or len(unclicked_candidate_ids) < args.negative_sampling_ratio:
                            continue # ! skip if row is not completed
                        # Drop if no clicked_news
                    candidate_news_ids = random.sample(clicked_candidate_ids, 1) + random.sample(unclicked_candidate_ids, args.negative_sampling_ratio)
                    if use_full_candidate:
                        candidate_news_ids = clicked_candidate_ids + unclicked_candidate_ids
                elif mode == 'test':
                    candidate_news_ids = literal_eval(row['candidate']) # Expected not empty list.

                if len(clicked_news_ids) < 1:
                    clicked_news_ids += [None] # padded with 1 empty news. # TODO for dynamic padding.
                    # ! Candidate news don't do this, because it expected to have news.

                clicked_title, clicked_abstract, clicked_category = get_grouped_news(clicked_news_ids)
                candidate_title, candidate_abstract, candidate_category = get_grouped_news(candidate_news_ids)

                clicked_news = {
                    'title': clicked_title,
                    'category': clicked_category
                    # 'abstract': clicked_abstract
                }
                
                candidate_news = {
                    'title': candidate_title,
                    'category': candidate_category
                    # 'abstract': candidate_abstract
                }
                if use_full_candidate:
                    result.append({
                        'user_id': row['user_id'],
                        'clicked_news_ids': clicked_news_ids,
                        'candidate_news_ids': candidate_news_ids,
                        'clicked_news': clicked_news,
                        'candidate_news': candidate_news,
                        'clicked': torch.tensor([1] * len(clicked_candidate_ids) + [0] * len(unclicked_candidate_ids), dtype=torch.float32)
                    })
                elif mode in ['train', 'valid']:
                    result.append({
                        'user_id': row['user_id'],
                        'clicked_news_ids': clicked_news_ids,
                        'candidate_news_ids': candidate_news_ids,
                        'clicked_news': clicked_news,
                        'candidate_news': candidate_news,
                        'clicked': torch.tensor([1] + [0] * args.negative_sampling_ratio, dtype=torch.float32)
                        # ! important for [RuntimeError: Expected floating point type for target with class probabilities, got Long]
                    })
                elif mode == 'test':
                    result.append({
                        'user_id': row['user_id'],
                        'clicked_news_ids': clicked_news_ids,
                        'candidate_news_ids': candidate_news_ids,
                        'clicked_news': clicked_news,
                        'candidate_news': candidate_news
                    })

            # Save
            torch.save(result, result_path)
            logging.info(f"{mode}.pt saved successfully at {result_path}.")
            self.result = result
    def __len__(self):
        return len(self.result)
    def __getitem__(self, index):
        return self.result[index]

@dataclass
class CustomDataCollator:
    tokenizer: CustomTokenizer
    mode: str
    use_full_candidate: bool
    def __call__(self, batch: list) -> dict[str, Any]:
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
        result = {
            'user': {
                'user_id': [example['user_id'] for example in batch],
                'clicked_news_ids': [example['clicked_news_ids'] for example in batch],
                'candidate_news_ids': [example['candidate_news_ids'] for example in batch],
                'clicked_news_category': [example['clicked_news']['category'] for example in batch],
                'candidate_news_category': [example['candidate_news']['category'] for example in batch]
            },
            'clicked_news': {
                'title': {
                    'input_ids': [],
                    'attention_mask': []
                },
                'category': []
            },
            'candidate_news': {
                'title': {
                    'input_ids': [],
                    'attention_mask': []
                },
            }
        }
        # Clicked & Clicked news category
        if self.use_full_candidate:
            max_clicked_len = max(len(example['clicked']) for example in batch)
            category_list = []
            clicked_list = []
            for example in batch:
                s = max_clicked_len - len(example['clicked'])
                category_padded = torch.full((s,), 0)
                clicked_padded = torch.full((s,), -1)
                category_list.append(torch.cat((example['candidate_news']['category'], category_padded)))
                clicked_list.append(torch.cat((example['clicked'], clicked_padded)))
            result['candidate_news']['category'] = torch.stack(category_list, dim=0)
            result['clicked'] = torch.stack(clicked_list, dim=0)
        elif self.mode in ['train', 'valid']:
            result['candidate_news']['category'] = torch.stack([example['candidate_news']['category'] for example in batch], dim=0)
            result['clicked'] = torch.stack([example['clicked'] for example in batch], dim=0) 
            # ! RuntimeError: cannot pin 'torch.cuda.LongTensor' only dense CPU tensors can be pinned
            # ! Don't need to move clicked and category to(device), do this in forward.

        if self.mode == 'train': # padded to same length
            lengths = [len(example['clicked_news_ids']) for example in batch]
            # print(sorted(lengths))
            # max_len = max(lengths)
            # mean_len = np.ceil(np.mean(lengths)).astype(int)
            # median_len = np.ceil(np.median(lengths)).astype(int)
            # percentile_length = int(np.percentile(lengths, 75))
            fixed_len = np.ceil(np.mean(lengths)).astype(int)
            for item in batch:
                # Clicked News
                encodes = item['clicked_news']['title'].copy() # encodes of title of clicked_news
                # ! 如果在 batching 的時候沒有用 shallow copy, 它會直接改變 dataset 的 value
                num_titles = len(encodes['input_ids'])
                category_list = item['clicked_news']['category'] # ! 是 tensor 所以不用 copy

                if num_titles < fixed_len:
                    padding_input_ids = torch.tensor([self.tokenizer.title_padding['input_ids']] * (fixed_len - num_titles))
                    padding_attention_mask = torch.tensor([self.tokenizer.title_padding['attention_mask']] * (fixed_len - num_titles))
                    # TODO, title_padding直接轉tensor
                    encodes['input_ids'] = torch.cat((encodes['input_ids'], padding_input_ids), dim=0)
                    encodes['attention_mask'] = torch.cat((encodes['attention_mask'], padding_attention_mask), dim=0)

                    padding_category = torch.tensor([0] * (fixed_len - num_titles))
                    category_list = torch.cat((category_list, padding_category), dim=0)

                elif num_titles > fixed_len:
                    encodes['input_ids'] = encodes['input_ids'][:fixed_len]
                    encodes['attention_mask'] = encodes['attention_mask'][:fixed_len]
                
                    category_list = category_list[:fixed_len]
                result['clicked_news']['title']['input_ids'].append(encodes['input_ids'])
                result['clicked_news']['title']['attention_mask'].append(encodes['attention_mask'])
                
                result['clicked_news']['category'].append(category_list)
                # Candidate News
                encodes = item['candidate_news']['title'] # encodes of title
                result['candidate_news']['title']['input_ids'].append(encodes['input_ids'])
                result['candidate_news']['title']['attention_mask'].append(encodes['attention_mask'])
        elif self.mode in ['valid', 'test']: # padded to dynamic length
            length = {
                'clicked_news': np.ceil(np.mean([len(example['clicked_news_ids']) for example in batch])).astype(int),
                'candidate_news': np.ceil(np.max([len(example['candidate_news_ids']) for example in batch])).astype(int)
            }
            category_fixed_len = length['clicked_news']
            for item in batch:
                # Category
                category_list = item['clicked_news']['category']
                category_len = len(category_list)
                if category_len < category_fixed_len:
                    padded = torch.tensor([0] * (category_fixed_len - category_len))
                    category_list = torch.cat((category_list, padded), dim=0)
                elif category_len > category_fixed_len:
                    category_list = category_list[:category_fixed_len]
                result['clicked_news']['category'].append(category_list)
                # Unlike training, candidate would be padded to max.
                for key in ['clicked_news', 'candidate_news']:
                    fixed_len = length[key]
                    # Clicked News & Candidate News
                    encodes = item[key]['title'].copy() # encodes of title of clicked_news
                    num_titles = len(encodes['input_ids'])
                    if num_titles < fixed_len:
                        padding_input_ids = torch.tensor([self.tokenizer.title_padding['input_ids']] * (fixed_len - num_titles))
                        padding_attention_mask = torch.tensor([self.tokenizer.title_padding['attention_mask']] * (fixed_len - num_titles))
                        # TODO, title_padding直接轉tensor
                        encodes['input_ids'] = torch.cat((encodes['input_ids'], padding_input_ids), dim=0)
                        encodes['attention_mask'] = torch.cat((encodes['attention_mask'], padding_attention_mask), dim=0)
                    elif num_titles > fixed_len:
                        encodes['input_ids'] = encodes['input_ids'][:fixed_len]
                        encodes['attention_mask'] = encodes['attention_mask'][:fixed_len]
                    result[key]['title']['input_ids'].append(encodes['input_ids'])
                    result[key]['title']['attention_mask'].append(encodes['attention_mask'])


        result['clicked_news']['title']['input_ids'] = torch.stack(result['clicked_news']['title']['input_ids'])
        result['clicked_news']['title']['attention_mask'] = torch.stack(result['clicked_news']['title']['attention_mask'])
        result['candidate_news']['title']['input_ids'] = torch.stack(result['candidate_news']['title']['input_ids'])
        result['candidate_news']['title']['attention_mask'] = torch.stack(result['candidate_news']['title']['attention_mask'])
        result['clicked_news']['category'] = torch.stack(result['clicked_news']['category'])
        return result

if __name__ == '__main__':

    pass