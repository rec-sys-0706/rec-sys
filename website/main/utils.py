import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import jwt
from werkzeug.security import generate_password_hash


load_dotenv()
SERVER_URL = os.environ.get('SERVER_URL')
BASE_URL = os.environ.get('BASE_URL')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
# Function to check environment variables
def check_env_vars():
    REQUIRED_ENV_VARS = ['SERVER_URL', 'BASE_URL', 'JWT_SECRET_KEY']
    missing_vars = [var for var in REQUIRED_ENV_VARS if os.getenv(var) is None]
    
    if missing_vars:
        missing_vars_str = '\n + '.join(missing_vars)
        raise ValueError(f"Missing environment variables:\n + {missing_vars_str}")
    else:
        print("All required environment variables are set.")
check_env_vars()



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
    response = requests.post(f'{SERVER_URL}/api/user/register', json = data)
    return response.content

#登入
def login(account, password):
    data = {
        "account" : f"{account}",
        "password" : f"{password}"
    }
    response = requests.post(f'{SERVER_URL}/api/user/login', json = data)
    response_json = json.loads(response.content)
    access_token = response_json.get('access_token')
    return access_token

#解碼
def access_decode(access_token):
    text = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=['HS256'])
    id = text.get('sub')
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    response = requests.get(f'{SERVER_URL}/api/user/{id}', headers=headers)
    return response.content

#修改user
def update_user_data(access_token, account, password, email, line_id):
    text = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=['HS256'])
    id = text.get('sub')
    headers = {
        "Authorization" : f'Bearer {access_token}'
    }
    password = generate_password_hash(password)
    get_user = {
        "account" : f"{account}",
        "password" : f"{password}",
        "email" : f"{email}",
        "line_id" : f"{line_id}"
    }
    requests.put(f"{SERVER_URL}/api/user/{id}", headers=headers, json=get_user)

def msg(text):
    string_data = text.decode('utf-8')
    data_dict = json.loads(string_data)
    message = data_dict.get('msg')
    if message == "Username already exists":
        message = 'exists'
    return message

def get_user(access_token):
    texts = access_decode(access_token)
    decoded_text = texts.decode('utf-8')
    json_data = json.loads(decoded_text)
    data = json_data['data']
    return data

def get_recommend(access_token):
    user = get_user(access_token)
    id = user['uuid']
    response = requests.get(f"{SERVER_URL}/api/user_history/recommend/{id}")
    data = response.json()
    items = []
    try:
        for entry in data:
            item_data = entry['item']
            item_data['recommendation_log_uuid'] = entry['recommendation_log_uuid']
            items.append(item_data)
        item = pd.DataFrame(items)
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = ''
    return item

def get_unrecommend(access_token):
    user = get_user(access_token)
    id = user['uuid']
    response = requests.get(f"{SERVER_URL}/api/user_history/unrecommend/{id}")
    data = response.json()
    items = []
    try:
        for entry in data:
            item_data = entry['item']
            item_data['recommendation_log_uuid'] = entry['recommendation_log_uuid']
            items.append(item_data)
        item = pd.DataFrame(items)
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = ''
    return item

def recommend_data_source(access_token, data_source):
    user = get_user(access_token)
    id = user['uuid']
    response = requests.get(f"{SERVER_URL}/api/user_history/recommend/{id}?data_source={data_source}")
    data = response.json()
    items = []
    try:
        for entry in data:
            item_data = entry['item']
            item_data['recommendation_log_uuid'] = entry['recommendation_log_uuid']
            items.append(item_data)
        item = pd.DataFrame(items)
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = ''
    return item

def unrecommend_data_source(access_token, data_source):
    user = get_user(access_token)
    id = user['uuid']
    response = requests.get(f"{SERVER_URL}/api/user_history/unrecommend/{id}?data_source={data_source}")
    data = response.json()
    items = []
    try:
        for entry in data:
            item_data = entry['item']
            item_data['recommendation_log_uuid'] = entry['recommendation_log_uuid']
            items.append(item_data)
        item = pd.DataFrame(items)
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = ''
    return item

def history_data_source(access_token, data_source):
    user = get_user(access_token)
    id = user['uuid']
    headers = {
        "Authorization" : f'Bearer {access_token}'
    }
    response = requests.get(f"{SERVER_URL}/api/user_history/{id}?data_source={data_source}", headers=headers)
    data = response.json()
    try:
        data = response.json()
        item = pd.json_normalize(data['history'])
        item['clicked_time'] = item['clicked_time'].apply(format_date)
        item['item_date'] = item['item_date'].apply(format_date)
    except:
        item = ''
    return item

def get_user_cliked(access_token):
    user = get_user(access_token)
    id = user['uuid']
    headers = {
        "Authorization" : f'Bearer {access_token}'
    }
    response = requests.get(f"{SERVER_URL}/api/user_history/{id}", headers=headers)
    try:
        data = response.json()
        item = pd.json_normalize(data['history'])
        item['clicked_time'] = item['clicked_time'].apply(format_date)
        item['item_date'] = item['item_date'].apply(format_date)
    except:
        item = ''
    return item

#獲取新聞
'''
def item_data():
    response = requests.get(f'{SERVER_URL}/api/item')
    items = response.json()
    
    item = items.sort_values('title')
    item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    return item
'''

def user_news(data_source):
    if data_source == 'all':
        data_source = ''
    response = requests.get(f'{SERVER_URL}/api/item/today?data_source={data_source}')
    data = response.json()
    item = pd.DataFrame(data)
    item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    return item

def get_formatted_datetime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def click_data(access_token, link):
    time = get_formatted_datetime()
    recommend_item = get_recommend(access_token)
    unrecommend_item = get_unrecommend(access_token)
    try:
        item_content = recommend_item.loc[recommend_item['link'] == link]
    except:
        item_content = unrecommend_item.loc[unrecommend_item['link'] == link]
    item_id = item_content['uuid'].iloc[0]
    uuid = item_content['recommendation_log_uuid'].iloc[0]
    user = get_user(access_token)
    id = user['uuid']
    data = {
        "user_id" : id,
        "item_id": item_id,
        "clicked_time": time
    }
    status = {
        "clicked": True
    }
    requests.put(f'{SERVER_URL}/api/recommend/{uuid}', json=status)
    requests.post(f'{SERVER_URL}/api/behavior', json = data)


def click_data_source(access_token, link, data_source):
    time = get_formatted_datetime()
    recommend_item = recommend_data_source(access_token, data_source)
    unrecommend_item = unrecommend_data_source(access_token, data_source)
    try:
        item_content = recommend_item.loc[recommend_item['link'] == link]
    except:
        item_content = unrecommend_item.loc[unrecommend_item['link'] == link]
    item_id = item_content['uuid'].iloc[0]
    uuid = item_content['recommendation_log_uuid'].iloc[0]
    user = get_user(access_token)
    id = user['uuid']
    data = {
        "user_id" : id,
        "item_id": item_id,
        "clicked_time": time
    }
    status = {
        "clicked": True
    }
    requests.put(f'{SERVER_URL}/api/recommend/{uuid}', json=status)
    requests.post(f'{SERVER_URL}/api/behavior', json = data)