import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from config import BaseConfig
import random
import pdb

class NewsDataset(Dataset):
    """
    News: {
        title (batch_size, num_tokens_title): tensor.
        title_mask: # TODO
    }
    For each element {

        clicked_news                                  : list[news], <num_clicked_news_a_user> sized list, each element is a News.
        candidate_news  (batch_size, )                : News. shape
        clicked         (batch_size, )                : int, 0/1 integer. shape
    }
    """
    def __init__(self,
                 config: BaseConfig,
                 behaviors_path,
                 news_path,
                 dest_dir=None) -> None:
        super().__init__()
        # ---- config ---- #
        self.num_tokens_title = config.num_tokens_title
        self.num_tokens_abstract = config.num_tokens_abstract
        self.num_clicked_news_a_user = config.num_clicked_news_a_user
        self.padding_news = {
            'title': torch.tensor([0] * self.num_tokens_title),
            'title_mask': torch.tensor([0] * self.num_tokens_title)
        }
        
        
        # ---------------- #
        self.__behaviors = pd.read_csv(behaviors_path,
                                       sep='\t')
        self.__news = pd.read_csv(news_path,
                                  sep='\t',
                                  index_col='news_id')
        self.result = self.__process_result(dest_dir)
        # else:
        #     temp = pd.read_csv(dest_dir + '/train.tsv',
        #                        sep='\t')
        #     temp['clicked_news'] = temp['clicked_news'].apply(literal_eval)
        #     temp['clicked_news_mask'] = temp['clicked_news_mask'].apply(literal_eval)
        #     temp['candidate_news'] = temp['candidate_news'].apply(literal_eval)
        #     temp['candidate_news_mask'] = temp['candidate_news_mask'].apply(literal_eval)
        #     self.result = temp.to_dict(orient='records')
        # TODO load from prepared.

    def __process_result(self, dest_dir):
        get_title = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title']))
        get_title_mask = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title_attention_mask']))

        result = []
        with tqdm(total=len(self.__behaviors)) as pbar:
            for _, row in self.__behaviors.iterrows():
                pbar.update(1)
                try:
                    clicked_news_ids = row['clicked_news'].split(' ')
                except Exception as e:
                    print(f'[ERROR] {row["user_id"]} Message: {e}')
                    clicked_news_ids = []
                
                for candidate_news_id, y in zip(row['candidate_news'].split(' '), row['clicked'].split(' ')):
                    random.shuffle(clicked_news_ids)
                    clicked_news_ids = clicked_news_ids[:self.num_clicked_news_a_user] # truncate
                    num_missing_news = self.num_clicked_news_a_user - len(clicked_news_ids)

                    # Clicked news
                    clicked_news = []
                    for news_id in clicked_news_ids:
                        news = {
                            'title': get_title(news_id),
                            'title_mask': get_title_mask(news_id)
                        }
                        clicked_news.append(news)
                    clicked_news += [self.padding_news] * num_missing_news

                    # Candidate news (only 1)
                    candidate_news = {
                        'title': get_title(candidate_news_id),
                        'title_mask': get_title_mask(candidate_news_id)
                    }

               
                    result.append({
                        'clicked_news': clicked_news,
                        'candidate_news': candidate_news,
                        'clicked': torch.tensor(int(y), dtype=torch.float32),
                    })
                if len(result) > 1000:
                    break # TODO
        # TODO save?
        # temp = pd.DataFrame(result)
        # temp.to_csv(dest_dir + '/train.tsv',
        #             sep='\t',
        #             index=False)
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
    dataset = NewsDataset(config, 'data/MINDsmall_train/behaviors_parsed.tsv', 'data/MINDsmall_train/news_parsed.tsv', 'data/MINDsmall_train')
    loader = DataLoader(dataset, batch_size=1)

    for item in loader:
        print(item['clicked_news'])
        break