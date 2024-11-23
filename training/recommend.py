"""
Recommend news for users.
call recommend function to get recommendations.
The project must contains
    1. test/tokenizer.json
    2. test/categorizer.json
    3. test/model.safetensors
"""
import random
import os
from typing import Any
from dataclasses import dataclass
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from safetensors.torch import load_file  # For loading .safetensors
import torch  # For saving .pt
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyodbc
from dotenv import load_dotenv

from training.model.NRMS import NRMS_BERT_test
from training.parameters import parse_args
from training.utils import CustomTokenizer, fix_all_seeds, list_to_dict

USER_KEYS = ['uuid']
ITEM_KEYS = ['uuid', 'title', 'category']
def dict_has_keys(dictionary: dict, required_keys: list):
    return all(key in dictionary for key in required_keys)

def generate_data(candidate_news_list: list[dict], user_list: list[dict], tokenizer: CustomTokenizer):
    # 載入 .env 文件中的環境變數
    load_dotenv()
    # 使用環境變數建立連接字串
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={os.environ.get('SQL_SERVER')};"
        f"DATABASE={os.environ.get('SQL_DATABASE')};"
        f"UID={os.environ.get('SQL_USERNAME')};"
        f"PWD={os.environ.get('SQL_PASSWORD')};"
    )

    # 建立資料庫連接
    try:
        conn = pyodbc.connect(connection_string)
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        raise

    # User Data
    user_data = []
    cursor = conn.cursor()
    for user in user_list:
        if not dict_has_keys(user, USER_KEYS): # Data Validation
            raise ValueError(f'Expected user has keys: {USER_KEYS}, but got: {user.keys()}')
        user_uuid = user['uuid']

        # 查詢用戶點擊過的新聞 (clicked_news)
        query = """
        SELECT
            behavior.item_id AS item_id,
            item.title AS title,
            item.category AS category
        FROM
            behavior
        INNER JOIN
            item ON behavior.item_id = item.uuid
        WHERE
            behavior.user_id = ?
        """

        # 修正執行查詢的方式，傳入參數作為元組
        cursor.execute(query, (user_uuid,))

        clicked_news_ids = []
        title = []
        category = []
        for row in cursor.fetchall():
            clicked_news_ids.append(row.item_id)
            title.append(row.title)
            category.append(tokenizer.encode_category(row.category))
        user_data.append({
            'uuid': user_uuid,
            'clicked_news_ids': clicked_news_ids if len(clicked_news_ids) else [-1],
            'clicked_news': {
                'title': tokenizer.encode_title(title) if len(title) else tokenizer.encode_title(['']),
                'category': category if len(category) else [0]
            }
        })

    # 關閉資料庫連接
    cursor.close()
    conn.close()

    # Item Data
    candidate_news = {
        'candidate_news_ids': [],
        'candidate_news': {
            'title': [],
            'category': []
        }
    }
    for example in candidate_news_list:
        if not dict_has_keys(example, ITEM_KEYS): # Data Validation
            raise ValueError(f'Expected item has keys: {ITEM_KEYS}, but got: {example.keys()}')
        candidate_news['candidate_news_ids'].append(example['uuid'])
        candidate_news['candidate_news']['title'].append(tokenizer.encode_title(example['title']))
        candidate_news['candidate_news']['category'].append(tokenizer.encode_category(example['category']))
    candidate_news['candidate_news']['title'] = list_to_dict(candidate_news['candidate_news']['title'])
    test_dataset = []
    for user in user_data:
        test_dataset.append({
            "user_id": user['uuid'],
            "clicked_news_ids": user['clicked_news_ids'],
            "candidate_news_ids": candidate_news['candidate_news_ids'],
            "clicked_news": user['clicked_news'],
            "candidate_news": candidate_news['candidate_news']
        })

    return test_dataset    

@dataclass
class TestDataCollator:
    tokenizer: CustomTokenizer
    def __call__(self, batch: list) -> dict[str, Any]:
        result = {
            'user': {
                'user_id': [example['user_id'] for example in batch],
                'clicked_news_ids': [example['clicked_news_ids'] for example in batch],
                'candidate_news_ids': [example['candidate_news_ids'] for example in batch],
                'clicked_news_category': [example['clicked_news']['category'] for example in batch],
                'candidate_news_category': [example['candidate_news']['category'] for example in batch]
            },
            "clicked_news": {
                "title": {
                    'input_ids': [],
                    'attention_mask': []
                }
            },
            'candidate_news': {
                'title': {
                    'input_ids': [],
                    'attention_mask': []
                },
                'category': []
            }
        }
        length = {
            'clicked_news': max(len(example['clicked_news_ids']) for example in batch),
            'candidate_news': max(len(example['candidate_news_ids']) for example in batch)
        }
        clicked_category_list = []
        candidate_category_list = []
        for example in batch:
            # Category
            if isinstance(example['clicked_news']['category'], list): # TODO optimize
                example['clicked_news']['category'] = torch.tensor(example['clicked_news']['category'], dtype=torch.int)
            if isinstance(example['candidate_news']['category'], list):
                example['candidate_news']['category'] = torch.tensor(example['candidate_news']['category'], dtype=torch.int)

            clicked_category_padded = torch.full((length['clicked_news'] - len(example['clicked_news_ids']), ), 0)
            clicked_category_list.append(torch.cat((example['clicked_news']['category'], clicked_category_padded)))
 
            candidate_category_padded = torch.full((length['candidate_news'] - len(example['candidate_news_ids']), ), 0)
            candidate_category_list.append(torch.cat((example['candidate_news']['category'], candidate_category_padded)))

            # Title
            for key in ['clicked_news', 'candidate_news']:
                fixed_len = length[key]
                encodes = example[key]['title'].copy()
                encodes['input_ids'] = torch.tensor(encodes['input_ids'], dtype=torch.int)
                encodes['attention_mask'] = torch.tensor(encodes['attention_mask'], dtype=torch.int)
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

        result['clicked_news']['category'] = torch.stack(clicked_category_list, dim=0)
        result['candidate_news']['category'] = torch.stack(candidate_category_list, dim=0)
        result['clicked_news']['title']['input_ids'] = torch.stack(result['clicked_news']['title']['input_ids'])
        result['clicked_news']['title']['attention_mask'] = torch.stack(result['clicked_news']['title']['attention_mask'])
        result['candidate_news']['title']['input_ids'] = torch.stack(result['candidate_news']['title']['input_ids'])
        result['candidate_news']['title']['attention_mask'] = torch.stack(result['candidate_news']['title']['attention_mask'])
        return result

def recommend(items: list[dict], users: list[dict]) -> list[dict]:
    """Recommend news for users.
    item: {
        uuid,
        title,
        category
    }
    user: {
        uuid,
    }
    """
    if len(items) == 0:
        raise ValueError('Items cannot be empty')
    if len(users) == 0:
        raise ValueError('Users cannot be empty')
    
    args = parse_args()
    args.model_name = 'NRMS-BERT'
    args.eval_batch_size = 2
    print(args)

    if not Path('./test/tokenizer.json').exists():
        raise FileNotFoundError('Tokenizer file not found')
    if not Path('./test/categorizer.json').exists():
        raise FileNotFoundError('Categorizer file not found')
    if not Path('./test/model.safetensors').exists():
        raise FileNotFoundError('Model file not found')
    fix_all_seeds(seed=args.seed)
    tokenizer = CustomTokenizer(args, './test/tokenizer.json', './test/categorizer.json')
    model = NRMS_BERT_test(args, tokenizer)
    model.load_state_dict(load_file("./test/model.safetensors"))


    collate_fn = TestDataCollator(tokenizer)
    test_dataset = generate_data(items, users, tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    model.eval()
    predictions = []
    user_ids = []
    clicked_news_ids = []
    candidate_news_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(**batch)
            predictions.append(outputs['logits'].cpu().numpy())
            user = outputs['user']
            user_ids += user['user_id']
            clicked_news_ids += user['clicked_news_ids']
            candidate_news_ids += user['candidate_news_ids']
    predictions = np.concatenate(predictions, axis=0)
    result = np.where(predictions > 0.5, 1, 0).tolist()

    num_clicked_news = [len(e) for e in clicked_news_ids]
    num_candidate_news = [len(e) for e in candidate_news_ids]
    df = pd.DataFrame({
        'user_id': user_ids,
        'clicked_news': clicked_news_ids,
        'candidate_news': candidate_news_ids,
        'predictions': [e[:l] for e, l in zip(predictions.tolist(), num_candidate_news)],
        'is_recommend': [e[:l] for e, l in zip(result, num_candidate_news)],
        'num_clicked_news': num_clicked_news,
        'num_candidate_news': num_candidate_news
    })
    df['clicked_news'] = df['clicked_news'].apply(lambda lst: [x for x in lst if x is not None])
    df = df.sort_values(by='user_id')
    
    user_to_account = pd.DataFrame(users).set_index('uuid')['account'].to_dict()
    item_to_title = pd.DataFrame(items).set_index('uuid')['title'].to_dict()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    recommendations = []
    for _, row in df.iterrows():
        for candidate_news_id, predict, is_recommend in zip(row.candidate_news, row.predictions, row.is_recommend):
            recommendations.append({
                'uuid': str(uuid4()),
                'user_id': row.user_id,
                'item_id': candidate_news_id,
                'recommend_score': predict,
                'is_recommend': is_recommend,
                'recommend_datetime': now
            })
            print(f'{predict}: ({user_to_account[row.user_id]}, {item_to_title[candidate_news_id]})')
    return recommendations

def generate_random_scores(items: list[dict], users: list[dict]) -> list[dict]:
    recommendations = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for user in users:
        user_uuid = user['uuid']
        for item in items:
            item_uuid = item['uuid']
            # tem_gattered_datetime = item['gattered_datetime']
            recommend_score = random.randint(0, 1)  
            recommendations.append({
                'uuid': str(uuid4()),
                'user_id': user_uuid,
                'item_id': item_uuid,
                'recommend_score': recommend_score,
                'is_recommend': recommend_score,
                'recommend_datetime': now
                # 'gattered_datetime': item_gattered_datetime,
                # 'clicked': 0
            })
    
    return recommendations