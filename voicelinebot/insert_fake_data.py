from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
from config import Config
from datetime import datetime

# 連接資料庫
DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

# 定義資料表
metadata = MetaData()
news_table = Table('news', metadata, autoload_with=engine)

# 假資料，不包含圖片
fake_news = [
    {
        "title": "最新科技新聞",
        "description": "這是一條關於最新科技的新聞內容。",
        "url": "https://example.com/news1",  # 這裡保留文章的 URL
        "published_at": datetime.now()  # 使用 datetime.now() 生成 datetime 對象
    },
    {
        "title": "體育新聞",
        "description": "這是一條關於體育的新聞內容。",
        "url": "https://example.com/news2",  # 這裡保留文章的 URL
        "published_at": datetime.now()
    },
    {
        "title": "娛樂新聞",
        "description": "這是一條關於娛樂的新聞內容。",
        "url": "https://example.com/news3",  # 這裡保留文章的 URL
        "published_at": datetime.now()
    }
]

# 插入假資料
for news in fake_news:
    insert_statement = news_table.insert().values(**news)
    session.execute(insert_statement)

session.commit()
session.close()
print("假資料插入成功！")
