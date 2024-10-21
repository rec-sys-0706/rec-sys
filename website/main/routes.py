from flask import Blueprint, render_template, send_file, request, session, redirect
from config import register, item_data, login, BASE_URL, click_data, user_data, update_user_data, msg
import matplotlib.pyplot as plt
import io
from PIL import Image
from collections import Counter
from datetime import date
import re
import pandas as pd

main_bp = Blueprint('main', 
                    __name__, 
                    template_folder='templates',
                    static_folder='static', 
                    url_prefix='/main')

# index 資料夾
@main_bp.route('/', methods = ['GET','POST'])
def index():
    status = 'T'
    if request.method == 'POST':
        account = request.form['account']
        password = request.form['password']
        msg = login(account, password)
        if msg == None:
            status = 'F'
        else:
            session['token'] = msg
            return render_template('./recommend/about.html')
    return render_template('./main/login.html', status = status)

def is_valid_email(email):
    #電子郵件格式
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    
    if re.fullmatch(pattern, email):
        return True
    else:
        return False

@main_bp.route('/signup', methods = ['GET','POST'])
def signup():
    status = 'T'
    if request.method == 'POST':
        email = request.form['email']
        account = request.form['account']
        password = request.form['password']

        if is_valid_email(email):
            status = 'True'
            text = register(email, account, password)
            message = msg(text)
            if(message == 'exists'):
                status = 'F'
        else:
            status = 'False' 
    return render_template('./main/signup.html', status = status)

# recommend 資料夾
@main_bp.route('/recommend')
def recommend():
    return render_template('./recommend/about.html')

@main_bp.route('/today_news', methods = ['GET','POST'])
def today_news():
    if 'token' in session:
        if request.method == 'POST':
            data = request.get_json()
            link = data.get('link')
            click_data(session['token'], link)
        all_news = item_data()
        today = date.today()
        today_time = today.strftime('%b %d, %Y')
        #news_date = all_news.loc[all_news['gattered_datetime'] == today_time]  正確的
        news_date = all_news.loc[all_news['gattered_datetime'] == 'May 31, 2024']
        return render_template('./recommend/today_news.html', all_news = news_date)
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/all_dates')
def all_dates():
    if 'token' in session:
        all_news = item_data()
        news_dates = all_news.sort_values('gattered_datetime').drop_duplicates(subset=['gattered_datetime'])
        return render_template('./recommend/all_dates.html', news_date = news_dates)        
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/all_news')
def allnews():
    if 'token' in session:
        all_news = item_data()
        date = request.args.get('gattered_datetime')
        date_news = all_news.loc[all_news['gattered_datetime'] == date]
        return render_template('./recommend/all_news.html', all_news = date_news)        
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/profile')
def profile():
    if 'token' in session:
        user = user_data(session['token'])
        return render_template('./recommend/profile.html', user_data = user)
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/revise', methods = ['GET','POST'])
def revise():
    if 'token' in session:
        user = user_data(session['token'])
    status = 'T'
    if request.method == 'POST':
        account = request.form['account']
        password = request.form['password']
        email = request.form['email']
        line_id = request.form['line_id']
        
        if is_valid_email(email):
            status = 'True'
            update_user_data(session['token'], account, password, email, line_id)
        else:
            status = 'False'
    return render_template('./recommend/revise.html', status = status, user_data = user)


@main_bp.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('./news.html', title=title, content=content)


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