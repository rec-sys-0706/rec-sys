import requests
import pandas as pd
import hmac
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import jwt

load_dotenv()
ROOT = os.environ.get('ROOT')
BASE_URL = os.environ.get('BASE_URL')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')


def get_signature(payload=''):
    # Get SQL_SECRET
    #secret_key = '123'
    secret_key = os.environ.get('SQL_SECRET')
    # Compute the HMAC-SHA256 signature
    hash_object = hmac.new(secret_key.encode('utf-8'), msg = payload.encode('utf-8'), digestmod=hashlib.sha256)
    signature = "sha256=" + hash_object.hexdigest()
    return signature
# payload = '{"example": "data"}' # 如果是 GET 則不用payload
# Prepare the headers, including the x-hub-signature-256

def format_date(date_str):
    return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z").strftime("%b %d, %Y")

#註冊
def register(email, account, password):
    data = {
        "account" : f"{account}",
        "password" : f"{password}",
        "email" : f"{email}",
        "line_id" : ""
    }
    response = requests.post(f'{ROOT}:5000/api/auth/register', json = data)
    return response.content

#登入
def login(account, password):
    data = {
        "account" : f"{account}",
        "password" : f"{password}"
    }
    response = requests.post(f'{ROOT}:5000/api/auth/login', json = data)
    response_json = json.loads(response.content)
    access_token = response_json.get('access_token')  
    return access_token

def access_decode(access_token):   
    text = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=['HS256'])
    id = text.get('sub')
    headers = {
        'Authorization': f'Bearer {access_token}',
        'X-Fju-Signature-256': get_signature()
    }
    response = requests.get(f'{ROOT}:5000/api/user/{id}', headers=headers)
    return response.content

#獲取新聞
def item_data(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'X-Fju-Signature-256': get_signature()
    }
    response = requests.get(f'{ROOT}:5000/api/item', headers = headers)
    items = response.json()
    try:
        items = pd.DataFrame(items['data'])
        item = items.sort_values('title')
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = 'error'
    return item

text = login('alice123', 'alice') # token

def user_data(access_token):
    texts = access_decode(access_token)
    decoded_text = texts.decode('utf-8')
    json_data = json.loads(decoded_text)
    data = json_data['data']
    user_data = pd.DataFrame([data])
    return user_data
'''
print(text)

item = item_data(text)
item_content = item.loc[item['link'] == 'https://www.oneusefulthing.org/p/the-lazy-tyranny-of-the-wait-calculation?utm_source=ai.briefnewsletter.com&utm_medium=referral&utm_campaign=chatgpt-plus-vs-copilot-pro']
item_id = item_content['uuid']
print(item_id)
'''

def get_formatted_datetime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def click_data(access_token, link):
    time = get_formatted_datetime()
    item = item_data(access_token)
    item_content = item.loc[item['link'] == link]
    item_id = item_content['uuid'].iloc[0]
    user = user_data(access_token)
    id = user['uuid'].iloc[0]
    data = {
        "user_id" : id,
        "item_id": item_id,
        "clicked_time": time
    }
    post = requests.post(f'{ROOT}:5000/api/user_history', json=data)
