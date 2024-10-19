import requests
import pandas as pd
import hmac
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid
import json

load_dotenv()
ROOT = os.environ.get('ROOT')


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

from werkzeug.security import generate_password_hash

#登入和註冊
def user_data(account, password, email):
    status=""
    if email != "":
        data = {
            "uuid": str(uuid.uuid4()),
            "account": "kevin134", 
            "password": generate_password_hash('kevin'), 
            "email": "kevin@example.com",
            "line_id": ""
        }
    else:
        data = {
            "account": account,
            "password": password
        }
        status = '/login'

    json_data = json.dumps(data)

    headers = {
        'Content-Type': 'application/json',
        'X-Fju-Signature-256': get_signature(json_data)
    }

    response = requests.post(f'{ROOT}:5000/api/user{status}', headers = headers, data = json_data)
    return response.text

#獲取新聞
def item_data():
    headers = {
        'Content-Type': 'application/json',
        'X-Fju-Signature-256': get_signature()
    }
    response = requests.get(f'{ROOT}:5000/api/item', headers = headers)
    items = response.json()
    item = pd.DataFrame(items['data'])
    item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    return item

#按照新聞title字母排列
def all():
    items = item_data()
    item = items.sort_values('title')
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



