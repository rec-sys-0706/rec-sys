import time
import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import jwt
from werkzeug.security import generate_password_hash
from flask import current_app

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



def time_since(start):
    return f'{round(time.time() - start, 2)}s'

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
    # TODO What if access token is empty or invalid?
    texts = access_decode(access_token)
    decoded_text = texts.decode('utf-8')
    json_data = json.loads(decoded_text)
    data = json_data['data']
    return data

# ---- get_document ---- #
def get_document(data_source, date):
    if data_source == 'all':
        data_source = ''
    print(f'Request data start.')
    start = time.time()
    response = requests.get(f'{SERVER_URL}/api/item/today?data_source={data_source}&date={date}')
    print(f'Request data end. Time taken: {time_since(start)}')
    data = response.json()
    return data

def get_document_for_user(access_token, data_source, is_recommend, date):
    user = get_user(access_token) # TODO 只存 uuid
    is_recommend = str(is_recommend).lower()
    response = requests.get(f"{SERVER_URL}/api/user_history/recommend/{user['uuid']}?data_source={data_source}&is_recommend={is_recommend}&date={date}")
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        response = requests.get(f'{SERVER_URL}/api/item/today?data_source={data_source}&date={date}')
        data = response.json()
        return data

def get_history(access_token, data_source):
    user = get_user(access_token)
    id = user['uuid']
    headers = {
        "Authorization" : f'Bearer {access_token}'
    }
    response = requests.get(f"{SERVER_URL}/api/user_history/{id}?data_source={data_source}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['history']
    else:
        print(f'Error: {response.status_code}')
        return []

def get_formatted_datetime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def click(access_token, item_id):
    user = get_user(access_token)
    id = user['uuid']
    time = get_formatted_datetime()
    clicked_data = {
        "user_id":  id,
        "item_id": item_id,
        "clicked_time": time
    }
    requests.post(f'{SERVER_URL}/api/behavior', json = clicked_data)