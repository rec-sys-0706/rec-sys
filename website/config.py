import requests
import pandas as pd
import hmac
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def get_signature(payload=''):
    # Get SQL_SECRET
    #secret_key = '123'
    secret_key = os.environ.get('SQL_SECRET')
    # Compute the HMAC-SHA256 signature
    signature = hmac.new(secret_key.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature    
# payload = '{"example": "data"}' # 如果是 GET 則不用payload
# Prepare the headers, including the x-hub-signature-256
headers = {
    'Content-Type': 'application/json',
    'x-fju-signature-256': f'sha256={get_signature()}'
}

news = pd.read_csv('./website/admin/data/news.tsv',
                    sep='\t',
                    names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                    index_col='news_id')

result = pd.read_csv('./website/admin/data/result.csv', index_col='user_id')

def format_date(date_str):
    return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z").strftime("%b %d, %Y")

response = requests.get('http://140.136.149.181:5000/api/news/', headers = headers)
test_news = pd.DataFrame(response.json())
test_news['date'] = test_news['date'].apply(format_date)