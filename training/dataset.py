import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from parameters import parse_args
import random
import logging
from pathlib import Path

def sample_users(consent_dict, k):


    # Ensure there is at least 1 accepted and k declined users available
    if len(accepted_users) < 1 or len(declined_users) < k:
        raise ValueError("Not enough users to sample from!")

    # Randomly select 1 accepted user
    selected_accepted = random.choice(accepted_users)

    # Randomly select k declined users
    selected_declined = random.sample(declined_users, k)

    # Combine the selected samples into a dictionary
    sample_dict = {selected_accepted: 1}
    sample_dict.update({user: 0 for user in selected_declined})

    return sample_dict
def sample(true_set: set[str], false_set: set[str], negative_sampling_ratio=4) -> tuple[str, str]:
    """negative sampling"""
    true_set = list(true_set)
    false_set = list(false_set)

    random.shuffle(true_set)
    random.shuffle(false_set)

    true_set = true_set[:1]
    false_set = false_set[:negative_sampling_ratio]

    news_id = true_set + false_set
    clicked = [1] * len(true_set) + [0] * len(false_set)

    if len(clicked) == (negative_sampling_ratio + 1):
        return (str(news_id), str(clicked))
    else:
        return None, None
    # # Create candidate_news & clicked columns
    # k = config.negative_sampling_ratio
    # behaviors[['candidate_news', 'clicked']] = [None, None]
    # for idx, row in tqdm(behaviors['impressions'].items(), total=len(behaviors)):
    #     true_set = set()
    #     false_set = set()
    #     counter = Counter()
    #     for e in row:
    #         news_id, clicked = e.split('-') # TODO assert size 2
    #         counter.update([news_id])
    #     filtered = {item: count for item, count in counter.items() if count > 1}
    #     if filtered:
    #         pdb.set_trace()
    #     for e in row:
    #         news_id, clicked = e.split('-') # TODO assert size 2
    #         true_set.add(news_id) if clicked == '1' else false_set.add(news_id)
    #     false_set -= true_set # Duplicated news_id is saved by true_set
    #     behaviors.loc[idx, ['candidate_news', 'clicked']] = sample(true_set, false_set, k) # TODO config
    # # behaviors = behaviors[behaviors['clicked'].apply(len) == 2*k+1] # Drop insufficient rows.
class NewsDataset(Dataset):
    """Sample data form `behaviors.tsv` and create dataset based on `news.tsv`

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
            data = torch.load(result_path)
            self.result, self.users = data['result'], data['users']
        else:
            logging.info(f"Cannot locate file {mode}.pt in '{src_dir}'.")
            logging.info(f"Starting data processing.")
            self.result, self.users = self.__process_result(result_path)

    def __process_result(self, filepath):
        get_title = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title']))
        get_title_mask = lambda news_id: torch.tensor(literal_eval(self.__news.loc[news_id]['title_attention_mask']))
    
        result = []
        users = []
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
                users.append({
                    'user_id': row['user_id'],
                    'clicked_news_ids': clicked_news_ids,
                    'candidate_news_ids': candidate_news_ids,
                    'labels': [int(y) for y in labels]
                })
            elif self.mode == 'test':
                result.append({
                    'clicked_news': clicked_news,
                    'candidate_news': candidate_news,
                })
                users.append({
                    'user_id': row['user_id'],
                    'clicked_news_ids': clicked_news_ids,
                    'candidate_news_ids': candidate_news_ids,
                })
        # Save
        torch.save({
            'result': result,
            'users': users
        }, filepath)
        logging.info(f"{self.mode}.pt saved successfully.")
        return result, users
    def __len__(self):
        return len(self.result)
    def __getitem__(self, index):
        """
        The input of model is (clicked_news, candidate_news, clicked)
        """
        return self.result[index]

if __name__ == '__main__':
    dataset = NewsDataset(BaseConfig(), mode='test')
