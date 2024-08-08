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
    For each element {
        clicked_news     : {
            title        : (batch_size, num_clicked_news_a_user, num_tokens_title)
            title_mask   : same as title
        }
        candidate_news   : {
            title        : (batch_size, 1 + k, num_tokens_title)
            title_mask   : same as title
        }
        clicked          : (batch_size, 1 + k)
    }
    """

    # list[news], <num_clicked_news_a_user> sized list, each element is a News.
    def __init__(self,
                 config: BaseConfig,
                 behaviors_path,
                 news_path) -> None:
        super().__init__()
        # ---- config ---- #
        self.num_tokens_title = config.num_tokens_title
        self.num_tokens_abstract = config.num_tokens_abstract
        self.num_clicked_news_a_user = config.num_clicked_news_a_user
        self.padding_title = torch.tensor([0] * self.num_tokens_title)
        # ---------------- #
        self.__behaviors = pd.read_csv(behaviors_path,
                                       sep='\t')
        self.__behaviors = self.__behaviors.dropna(subset=['clicked_news']) # TODO If no clicked_news, should it be dropped?
        self.__news = pd.read_csv(news_path,
                                  sep='\t',
                                  index_col='news_id')
        
        train_filepath = Path(behaviors_path).parent / 'train.pt'
        if not config.is_new:
            assert train_filepath.exists(), 'train.pt does not exist.'
            self.result = torch.load(train_filepath)
        else:
            self.result = self.__process_result(train_filepath)

    def __process_result(self, train_filepath):
        get_title = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title']))
        get_title_mask = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title_attention_mask']))
        print(self.__behaviors[self.__behaviors['clicked_news'].isna()])
        result = []
        for _, row in tqdm(self.__behaviors.iterrows(), total=len(self.__behaviors)):
            clicked_news_ids = row['clicked_news'].split(' ')
            candidate_news_ids = row['candidate_news'].split(' ')
            labels = row['clicked'].split(' ')
            # Clicked news
            random.shuffle(clicked_news_ids)
            clicked_news_ids = clicked_news_ids[:self.num_clicked_news_a_user] # truncate
            num_missing_news = self.num_clicked_news_a_user - len(clicked_news_ids)

            clicked_news = {
                'title': torch.stack([get_title(news_id) for news_id in clicked_news_ids] + [self.padding_title] * num_missing_news),
                'title_mask': torch.stack([get_title_mask(news_id) for news_id in clicked_news_ids] + [self.padding_title] * num_missing_news)
            }
            # Candidate news
            candidate_news = {
                'title': torch.stack([get_title(news_id) for news_id in candidate_news_ids]),
                'title_mask': torch.stack([get_title_mask(news_id) for news_id in candidate_news_ids])
            }
            
            result.append({
                'clicked_news': clicked_news,
                'candidate_news': candidate_news,
                'clicked': torch.tensor([int(y) for y in labels], dtype=torch.float32),
            })

        # Save
        torch.save(result, train_filepath)
        logging.info('train.csv file saved successfully.')
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
    dataset = NewsDataset(config, 'data/MINDsmall_train/behaviors_parsed.tsv', 'data/MINDsmall_train/news_parsed.tsv')
    loader = DataLoader(dataset, batch_size=2)

    for item in loader:
        x1 = item['clicked_news']
        x2 = item['candidate_news']
        y = item['clicked']
        pdb.set_trace()