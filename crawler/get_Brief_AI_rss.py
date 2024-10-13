import feedparser
from bs4 import BeautifulSoup
import csv
import os

def fetch_and_update_rss(rss_url, csv_file):
    """
    從指定的 RSS URL 獲取資料，並更新至 CSV 檔案，避免重複資料。

    rss_url: RSS feed 的 URL
    csv_file: 要寫入的 CSV 檔案名稱
    """
    
    feed = feedparser.parse(rss_url)

    #檢查csv檔是否存在，以及讀取資料以便檢查資料是否重複
    existing_data = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  
            for row in reader:
                existing_data.add((row[0], row[2]))  

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
            writer.writerow(["Title", "Published Date", "Link", "Content"])

        #抓取rss內容
        for entry in feed.entries:
            title = entry.title
            published_date = entry.published
            link = entry.link

            content = ""
            if 'content' in entry:
                raw_html = entry.content[0].value
                soup = BeautifulSoup(raw_html, "html.parser")
                content = soup.get_text()

            if (title, link) not in existing_data:
                writer.writerow([title, published_date, link, content])
                existing_data.add((title, link))

    print("更新成功")

rss_url = "https://rss.beehiiv.com/feeds/kdQL8Qkqgp.xml"
csv_file = "rss_feed.csv"
fetch_and_update_rss(rss_url, csv_file)
