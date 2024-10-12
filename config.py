import os
from flask_sqlalchemy import SQLAlchemy


DB = SQLAlchemy()

class Config:
    DRIVER = 'ODBC Driver 17 for SQL Server'
    SERVER = os.environ.get('SQL_SERVER')
    DATABASE = os.environ.get('SQL_DATABASE')
    USERNAME = os.environ.get('SQL_USERNAME')
    PASSWORD = os.environ.get('SQL_PASSWORD')
    SQLALCHEMY_DATABASE_URI = (
        f'mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}:{1433}/{DATABASE}?'
        f'driver={DRIVER}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False