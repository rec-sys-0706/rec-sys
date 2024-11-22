import pandas as pd
from crawler.get_cnn_news import scrape_cnn_articles
from crawler.convert_category import convert_category
from crawler.get_daily_papers import scrape_huggingface_papers
from crawler.insret_item_and_score_test import post_news_and_score
from crawler.insert_daily_papers_test import post_papers_and_score
from crawler.get_users import get_users

papers_file = scrape_huggingface_papers()

cnn_file = scrape_cnn_articles()

if cnn_file:
    df = pd.read_csv(cnn_file)
    df = convert_category(df)
    df.to_csv(cnn_file, index=False)
    print("整理成功")
else:
    print("未產生新檔案。")
    
users = get_users()

post_papers_and_score(papers_file, users)

post_news_and_score(cnn_file, users)

print("定期爬蟲結束")
    



