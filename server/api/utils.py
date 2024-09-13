import pyodbc
import os
import hashlib
import hmac

def get_db_connection():
    server = os.environ.get('SQL_SERVER')
    database = os.environ.get('SQL_DATABASE')
    username = os.environ.get('SQL_USERNAME')
    password = os.environ.get('SQL_PASSWORD')
    driver = os.environ.get('SQL_DRIVER')

    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'
    conn = pyodbc.connect(conn_str)
    return conn

def check_api_key(req) -> bool:
    secret = '123' # os.environ.get('SQL_SECRET')
    signature = req.headers.get('X-Fju-Signature-256')
    payload = req.data

    if signature is None:
        print("No signature is provided.")
        return False

    hash_object = hmac.new(secret.encode('utf-8'), msg=payload, digestmod=hashlib.sha256)
    expected_signature = "sha256=" + hash_object.hexdigest()

    return hmac.compare_digest(expected_signature, signature)
