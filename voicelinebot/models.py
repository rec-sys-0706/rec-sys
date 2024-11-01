import sqlalchemy as db
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from config import Config

# 創建資料庫引擎和元數據
engine = db.create_engine(Config.SQLALCHEMY_DATABASE_URI)
metadata = db.MetaData()

# 定義資料表結構
news_table = db.Table(
    'news', metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String, nullable=False),
    Column('description', String),
    Column('url', String),
    Column('image_url', String),
    Column('published_at', DateTime, default=datetime.now)
)

user_actions_table = db.Table(
    'user_actions', metadata,
    Column('id', Integer, primary_key=True),
    Column('user_id', String, nullable=False),
    Column('news_id', Integer, nullable=False),
    Column('action_type', String, nullable=False),
    Column('action_time', DateTime, default=datetime.now)
)

# 建立資料表
metadata.create_all(engine)
