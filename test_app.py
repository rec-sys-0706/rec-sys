import hashlib
import hmac
import secrets
from flask import Flask
import requests
import os

register_data = {
    "account" : "test111",
    "password" : "12345678",
    "email" : "test111@example.com",
}

register_data = requests.post(f"{os.environ.get('ROOT')}:5000/api/auth/register", json=register_data)
print(register_data.content)

# login_data = {
#     "account" : "test111",
#     "password" : "12345678"
# }

# login_data = requests.post(f"{os.environ.get('ROOT')}:5000/api/auth/login", json=login_data)
# print(login_data.content)
