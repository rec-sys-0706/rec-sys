from flask import Flask, render_template, send_file, request
# from config import test_news
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
from collections import Counter
import datetime
import re

app = Flask(__name__)


# TODO
data = {
    'title': ['News 1', 'News 2', 'News 3', 'News 4'],
    'category': ['World', 'Sports', 'Entertainment', 'Technology'],
    'date': ['Oct 06, 2024', 'Oct 05, 2024', 'Oct 04, 2024', 'Oct 06, 2024'],
    'description': ['This is the first news', 'This is the second news', 'This is the third news', 'This is the fourth news']
}
test_news = pd.DataFrame(data)

@app.route('/')
def index():
    return render_template('index.html')

# recommend 資料夾
@app.route('/recommend', methods=['GET'])
def recommend():
    return render_template('./recommend/aboutus.html')

news_dates = test_news.sort_values('date').drop_duplicates(subset=['date'])
all_news = test_news.sort_values('title')
 
@app.route('/main/today_news')
def today_news():
    today = datetime.date.today()
    today_time = today.strftime('%b %d, %Y')
    return render_template('./recommend/today_news.html', news_date=news_dates, 
                           today_time=today_time, all_news=all_news)
    
@app.route('/main/all_dates')
def all_dates():
    return render_template('./recommend/all_dates.html', news_date = news_dates)

@app.route('/main/all_news')
def all_news_view():
    # 獲取不重複的日期
    news_dates = all_news['date'].unique()

    return render_template('./recommend/all_news.html', news_date=news_dates, all_news=all_news)

@app.route('/main/aboutus')
def abotus():
    return render_template('recommend/aboutus.html')

@app.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('news.html', title=title, content=content)

# TODO
# Sample news data
articles = [
    {
        'category': 'Breaking News',
        'title': 'Breaking News: Market Hits All-Time High',
        'content': 'The stock market has reached an all-time high today, with major indices showing significant gains.',
        'author': 'John Doe',
        'date': '2024-08-09'
    },
    {
        'category': 'Tech Innovations',
        'title': 'Tech Innovations: AI Revolutionizing Industries',
        'content': 'Artificial Intelligence is transforming the way businesses operate, from automation to customer service.',
        'author': 'Jane Smith',
        'date': '2024-08-08'
    },
    {
        'category': 'Breaking News',
        'title': 'Breaking News: Market Hits All-Time High',
        'content': 'The stock market has reached an all-time high today, with major indices showing significant gains.',
        'author': 'John Doe',
        'date': '2024-08-09'
    },
    {
        'category': 'Tech Innovations',
        'title': 'Tech Innovations: AI Revolutionizing Industries',
        'content': 'Artificial Intelligence is transforming the way businesses operate, from automation to customer service.',
        'author': 'Jane Smith',
        'date': '2024-08-08'
    },
    {
        'category': 'Health Update',
        'title': 'Health Update: New Breakthrough in Cancer Research',
        'content': 'Scientists have announced a major breakthrough in cancer treatment, promising better outcomes for patients.',
        'author': 'Alice Brown',
        'date': '2024-08-07'
    },
]

if __name__ == '__main__':
    app.run(debug=True)
