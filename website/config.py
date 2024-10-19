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

'''
texts = access_decode(text)
decoded_text = texts.decode('utf-8')
json_data = json.loads(decoded_text)

data = json_data['data']

df = pd.DataFrame([data])

print(df)
'''
'''
text = login('alice123', 'alice')
if text == None:
    print(1)
else:
    print(2)

response_json = json.loads(texts)
access_token = response_json.get('access_token')
print("Access Token:", access_token) #要得

text = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=['HS256'])
print(text)
account = text.get('sub')
print(account)
'''

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

'''
user = {
    "account": "kevin134",
    "password": "kevin"
}

user_data = json.dumps(user)

header = {
    'Content-Type': 'application/json',
    'X-Fju-Signature-256': get_signature(user_data)
}
#response = requests.post(f'{ROOT}:5000/api/user', headers = header, data=json_data)
response = requests.post(f'{ROOT}:5000/api/user/login', headers = header, data = user_data)

if response.status_code == 201:
    print("資料新增成功")
    print("伺服器回應：", response.json())
else:
    print(f"發生錯誤，狀態碼: {response.status_code}")
    print("錯誤訊息：", response.text)

'''

'''
test_news = response.json()

test_new = pd.DataFrame(test_news['data'])
test_new['gattered_datetime'] = test_new['gattered_datetime'].apply(format_date)
#print(test_new['gattered_datetime'])
print(test_new['title'])
'''



