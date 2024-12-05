import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from marshmallow import fields, ValidationError

# Define a custom DateTime field
class CustomDateTime(fields.DateTime):
    def __init__(self, format='%Y-%m-%dT%H:%M:%S', required=True, error_messages=None, **kwargs):
        # Define default error messages if not provided
        if error_messages is None:
            error_messages = {
                "required": "This field is required.",
                "invalid": "Invalid date format."
            }
        super().__init__(format=format, required=required, error_messages=error_messages, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return super()._deserialize(value, attr, data, **kwargs)
        except ValidationError:
            raise ValidationError(self.error_messages["invalid"])

class Base(DeclarativeBase):
    pass

DB = SQLAlchemy(model_class=Base)
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
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'super_secret')

    # Flask-Compress 配置
    COMPRESS_MIMETYPES = ['text/html', 'text/css', 'application/json', 'application/javascript']
    COMPRESS_LEVEL = 6  # 壓縮等級（1-9）
    COMPRESS_MIN_SIZE = 500  # 最小壓縮大小（字節）