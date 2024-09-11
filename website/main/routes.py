from flask import Blueprint, render_template, send_file
from config import test_news
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
from collections import Counter
import datetime


main_bp = Blueprint('main', 
                    __name__, 
                    template_folder='templates',
                    static_folder='static', 
                    url_prefix='/main')

@main_bp.route('/')
def home():
    return render_template('index.html')

@main_bp.route('/recommend')
def recommend():
    news_date = test_news.sort_values('date').drop_duplicates(subset=['date'])
    all_news = test_news.sort_values('title')
    today = datetime.date.today()
    today_time = today.strftime('%b %d, %Y') 
    print(today_time)
    return render_template('recommend.html', news_date = news_date, all_news = all_news, today_time = today_time, user_info = user_info)

@main_bp.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('news.html', title=title, content=content)

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

user_info = {
        'Account': 'John',
        'password': 'xxxxxxxxx',
        'email': 'johndoe@example.com',
        'phone': '123-456-7890'   
}

@main_bp.route('/donut_chart.png')
def donut_chart():
    categories = [article['category'] for article in articles]
    category_counts = Counter(categories)

    labels = category_counts.keys()
    sizes = category_counts.values()

    # Create a pie chart with a hole in the center (donut chart)
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.4})

    # Draw a circle in the center to make it a donut chart
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    # Ensure the chart is a circle
    ax.axis('equal')

    # Save the chart to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')