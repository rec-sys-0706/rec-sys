from flask import Flask
import requests
import os

fake_item = {

    "title": "Understanding AI and Its Applications",
    "abstract": "This article provides an in-depth understanding of artificial intelligence and its applications across various industries.",
    "link": "https://example.com/ai-applications",
    "data_source": "cnn_news",
    "gattered_datetime": "2024-10-20T08:15:00"
}

# fake_item = requests.post(f"{os.environ.get('ROOT')}:5000/api/item/crawler", json=fake_item)
# print(fake_item)

today_news = requests.get(f"{os.environ.get('ROOT')}:5000/api/item/today")
for item in today_news.json():
    print(item['date'])

