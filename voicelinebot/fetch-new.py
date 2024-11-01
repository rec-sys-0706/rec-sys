import requests
import sqlalchemy as db
from config import NEWS_API_URL, DATABASE_URI
from datetime import datetime

# 初始化資料庫連接
engine = db.create_engine(DATABASE_URI)
connection = engine.connect()
metadata = db.MetaData()
news_table = db.Table('news', metadata, autoload=True, autoload_with=engine)

def fetch_news():
    response = requests.get(NEWS_API_URL)
    news_data = response.json()

    for article in news_data["articles"]:
        title = article["title"]
        description = article["description"]
        url = article["url"]
        image_url = article.get("urlToImage", "")
        published_at = article["publishedAt"]

        # 插入到資料庫
        insert = news_table.insert().values(
            title=title,
            description=description,
            url=url,
            image_url=image_url,
            published_at=published_at,
            created_at=datetime.now()
        )
        connection.execute(insert)

# 每天自動執行
if __name__ == "__main__":
    fetch_news()
