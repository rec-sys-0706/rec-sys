from flask import Blueprint, render_template
from config import test_news
import datetime

main_bp = Blueprint('main', 
                    __name__, 
                    template_folder='templates',
                    static_folder='static', 
                    url_prefix='/main')

@main_bp.route('/')
def home():
    news_date = test_news.sort_values('date').drop_duplicates(subset=['date'])
    news = test_news.sort_values('title')
    return render_template('index.html', news_date = news_date, news_article = news)

@main_bp.route('/recommend')
def recommend():
    news_date = test_news.sort_values('date').drop_duplicates(subset=['date'])
    news = test_news.sort_values('title')
    return render_template('recommend.html', news_date = news_date, news_article = news)

@main_bp.route('/recommend')
def recommend():
    news_date = test_news.sort_values('date').drop_duplicates(subset=['date'])
    all_news = test_news.sort_values('title')
    today = datetime.date.today()
    #today_time = today.strftime('%b %d, %Y') 
    today_time = today.strftime('%b 01, %Y') 
    print(today_time)
    return render_template('recommend.html', news_date = news_date, all_news = all_news, today_time = today_time)

@main_bp.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('news.html', title=title, content=content)
