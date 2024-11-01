from flask import Blueprint, render_template, request, session, redirect
from config import register, item_data, login, BASE_URL, click_data, user_data, update_user_data, msg, get_recommend, get_unrecommend, get_user_cliked
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
        recommend_new = get_recommend(session['token'])
        unrecommend_news = get_unrecommend(session['token'])
        return render_template('./recommend/today_news.html', recommend_new = recommend_new, unrecommend_news = unrecommend_news)
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
        history = get_user_cliked(session['token'])
        return render_template('./recommend/profile.html', user_data = user, history = history)
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/revise', methods = ['GET','POST'])
def revise():
    if 'token' in session:
        user = user_data(session['token'])
        history = get_user_cliked(session['token'])
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
    return render_template('./recommend/revise.html', status = status, user_data = user, history = history)


@main_bp.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('./news.html', title=title, content=content)