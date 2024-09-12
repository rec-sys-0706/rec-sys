import pyodbc
import os

def env_var(key, default_value=None):
    return os.environ.get(key, default_value)

def get_db_connection():
    server = env_var('SQL_SERVER')
    database = env_var('SQL_DATABASE')
    username = env_var('SQL_USERNAME')
    password = env_var('SQL_PASSWORD')
    driver = env_var('SQL_DRIVER')

    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'
    conn = pyodbc.connect(conn_str)
    return conn

def check_api_key(req):
    pass