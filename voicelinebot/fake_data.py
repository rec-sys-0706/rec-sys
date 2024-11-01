from models import engine, news_table
from sqlalchemy import insert
from datetime import datetime

# 連接到資料庫
connection = engine.connect()

# 插入假新聞資料
def insert_fake_news():
    fake_news = [
        {'title': '假新聞標題 1', 'description': '這是一條假新聞', 'url': 'http://example.com/1', 'image_url': 'http://example.com/image1.jpg', 'published_at': datetime.now()},
        {'title': '假新聞標題 2', 'description': '這是一條假新聞', 'url': 'http://example.com/2', 'image_url': 'http://example.com/image2.jpg', 'published_at': datetime.now()},
        {'title': '假新聞標題 3', 'description': '這是一條假新聞', 'url': 'http://example.com/3', 'image_url': 'http://example.com/image3.jpg', 'published_at': datetime.now()},
    ]
    connection.execute(insert(news_table), fake_news)

# 插入假新聞
insert_fake_news()
