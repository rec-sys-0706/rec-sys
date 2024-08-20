import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from config import BaseConfig
import random
import logging
from pathlib import Path
import pdb

class NewsDataset(Dataset):
    """

    If mode=='train', it
    For each element {
        clicked_news       : {
            title          : (batch_size, num_clicked_news_a_user, num_tokens_title)
            title_mask     : same as title
        }
        candidate_news     : {
            title          : (batch_size, 1 + k, num_tokens_title)
            title_mask     : same as title
        }
        clicked(valid/test): (batch_size, 1 + k)
    }
    """

    # list[news], <num_clicked_news_a_user> sized list, each element is a News.
    def __init__(self,
                 config: BaseConfig,
                 mode) -> None:
        super().__init__()
        # ---- config ---- #
        self.mode = mode
        self.num_tokens_title = config.num_tokens_title
        self.num_tokens_abstract = config.num_tokens_abstract
        self.num_clicked_news_a_user = config.num_clicked_news_a_user
        self.padding_title = torch.tensor([0] * self.num_tokens_title)
        # ---------------- #
        if mode == 'train':
            src_dir = Path(config.train_dir)
        elif mode == 'valid':
            src_dir = Path(config.val_dir)
        elif mode == 'test':
            src_dir = Path(config.test_dir)
        else:
            raise ValueError(f"[ERROR] Expected 'mode' be str['train'|'valid'|'test'] but got '{mode}'.")
        behaviors_path = src_dir / 'behaviors_parsed.tsv'
        news_path = src_dir / 'news_parsed.tsv'
        if not behaviors_path.exists():
            raise FileNotFoundError(f"Cannot locate behaviors_parsed.tsv file in '{src_dir}'")
        elif not news_path.exists():
            raise FileNotFoundError(f"Cannot locate news_parsed.tsv file in '{src_dir}'")

        self.__behaviors = pd.read_csv(behaviors_path,
                                       sep='\t')
        self.__behaviors = self.__behaviors.dropna(subset=['clicked_news']) # TODO If no clicked_news, should it be dropped?
        self.__news = pd.read_csv(news_path,
                                  sep='\t',
                                  index_col='news_id')
        
        result_path = src_dir / f'{mode}.pt' 
        if result_path.exists():
            self.result = torch.load(result_path)
        else:
            logging.info(f"Cannot locate file {mode}.pt in '{src_dir}'.")
            logging.info(f"Starting data processing.")
            self.result = self.__process_result(result_path)

    def __process_result(self, filepath):
        get_title = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title']))
        get_title_mask = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title_attention_mask']))
    
        result = []
        for _, row in tqdm(self.__behaviors.iterrows(), total=len(self.__behaviors)):
            # Clicked news
            clicked_news_ids = row['clicked_news'].split(' ')
            random.shuffle(clicked_news_ids)
            clicked_news_ids = clicked_news_ids[:self.num_clicked_news_a_user] # truncate
            num_missing_news = self.num_clicked_news_a_user - len(clicked_news_ids)

            clicked_news = {
                'title': torch.stack([get_title(news_id) for news_id in clicked_news_ids] + [self.padding_title] * num_missing_news),
                'title_mask': torch.stack([get_title_mask(news_id) for news_id in clicked_news_ids] + [self.padding_title] * num_missing_news)
            }
            # Candidate news
            candidate_news_ids = row['candidate_news'].split(' ')
            candidate_news = {
                'title': torch.stack([get_title(news_id) for news_id in candidate_news_ids]),
                'title_mask': torch.stack([get_title_mask(news_id) for news_id in candidate_news_ids])
            }
            
            if self.mode in {'train', 'valid'}:
                labels = row['clicked'].split(' ')
                result.append({
                    'clicked_news': clicked_news,
                    'candidate_news': candidate_news,
                    'clicked': torch.tensor([int(y) for y in labels], dtype=torch.float32),
                })
            elif self.mode == 'test':
                result.append({
                    'clicked_news': clicked_news,
                    'candidate_news': candidate_news,
                })

        # Save
        torch.save(result, filepath)
        logging.info(f"{self.mode}.pt saved successfully.")
        return result
    def __len__(self):
        return len(self.result)
    def __getitem__(self, index):
        """
        The input of model is (clicked_news, candidate_news, clicked)
        """
        return self.result[index]

if __name__ == '__main__':


    config = BaseConfig()
    dataset = NewsDataset(config, 'valid')
    loader = DataLoader(dataset, batch_size=2)

    for item in loader:
        x1 = item['clicked_news']
        x2 = item['candidate_news']
        pdb.set_trace()
        y = item['clicked']
        pdb.set_trace()