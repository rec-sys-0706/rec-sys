from get_cnn_news import scrape_cnn_articles
from convert_category import convert_category
from get_daily_papers import scrape_huggingface_papers
from insret_item_and_score import post_news_and_score
import pandas as pd

papers_file = scrape_huggingface_papers()

cnn_file = scrape_cnn_articles()

if cnn_file:
    df = pd.read_csv(cnn_file)
    df = convert_category(df)
    df.to_csv(cnn_file, index=False)
    print("整理成功")
else:
    print("未產生新檔案。")
    



