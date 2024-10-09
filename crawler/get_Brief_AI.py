import feedparser
from bs4 import BeautifulSoup

rss_url = "https://rss.beehiiv.com/feeds/kdQL8Qkqgp.xml"

feed = feedparser.parse(rss_url)

for entry in feed.entries:
    print(f"Title: {entry.title}")
    print(f"Published Date: {entry.published}")
    print(f"Link: {entry.link}")
    
    if 'content' in entry:
        raw_html = entry.content[0].value
        soup = BeautifulSoup(raw_html, "html.parser")
        plain_text = soup.get_text()
        
        print(f"Content (text only): {plain_text}")
        
    print("---")
