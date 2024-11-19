import requests
import pandas as pd
import hmac
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
ROOT = os.environ.get('ROOT')
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

def format_date(date_str):
    return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z").strftime("%b %d, %Y")

response = requests.get(f'{ROOT}:5000/api/news/', headers = headers)
    
test_news = pd.DataFrame(response.json())
test_news['date'] = test_news['date'].apply(format_date)


# 測試
data = {
    "account": "alice123",
    "password": "alice"
}

'''
responses = requests.get(f'{ROOT}:5000/api/user/verification', headers = headers, json=data)

responses = requests.delete(f'{ROOT}:5000/api/reader_record/7', headers = headers)

try:
    data = responses.json()
    print(data['message'])
    if isinstance(data, dict):  # Single user (a dictionary)
        # Convert the single user data into a DataFrame
        user = pd.DataFrame([data])  # Wrap the dictionary in a list
        print(user)
    elif isinstance(data, list):  # Multiple users (a list)
        users = pd.DataFrame(data)
        print(users)
    else:
        print("Unexpected data format:", data)
except ValueError as e:
    print("Error while parsing JSON:", e)
    
'''