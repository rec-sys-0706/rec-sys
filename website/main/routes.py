import os
from flask import Blueprint, render_template, request, session, redirect, current_app
from .utils import register, login, click_data, get_user, update_user_data, msg, get_recommend, get_unrecommend, get_user_cliked, recommend_data_source, unrecommend_data_source, history_data_source, click_data_source, user_news
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
@main_bp.route('/login', methods = ['GET','POST'])
def login_user():
    status = 'T'
    if session.get('token'):
        return redirect('/main')
    if request.method == 'POST':
        account = request.form['account']
        password = request.form['password']
        msg = login(account, password)
        if msg == None:
            status = 'F'
        else:
            session['token'] = msg
            session['page'] = 'recommend'
            session['source'] = 'all'
            session['account'] = account
            return redirect('/main')
    return render_template('./auth/login.html', status = status)

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
    return render_template('./auth/signup.html', status = status)

# recommend 資料夾
@main_bp.route('/recommend', methods = ['GET','POST'])
def recommend():
    session['page'] = 'recommend'
    try:
        data = request.get_json()
        source = data.get('source')
        session['source'] = source
    except:
        print('error')
    return render_template('./main/about.html')

@main_bp.route('/', methods = ['GET','POST'])
def index():
    session['page'] = ''
    if 'token' in session and session['token'] != '':
        status = 'Login'
        if request.method == 'POST':
            try:                    
                data = request.get_json()
                link = data.get('link')
                current_app.logger.info(session)
                if session['source'] == 'all':
                    click_data(session['token'], link)
                else:
                    click_data_source(session['token'], link, session['source'])
            except:
                data = request.get_json()
                source = data.get('source')
                session['source'] = source
        if session['source'] == 'all':
            recommend_news = get_recommend(session['token'])
            unrecommend_news = get_unrecommend(session['token'])
        else:
            recommend_news = recommend_data_source(session['token'], session['source'])
            unrecommend_news = unrecommend_data_source(session['token'], session['source']) 
        return render_template('./main/today_news.html', status = status, recommend_news = recommend_news, unrecommend_news = unrecommend_news)
    else:
        recommend_news = ''
        session['token'] = ''
        try:
            data = request.get_json()
            source = data.get('source')
            session['source'] = source
        except:
            pass
        if 'source' in session:
            current_app.logger.info(session)
            unrecommend_news = user_news(session['source']) 
        else:
            session['source'] = 'all'
            unrecommend_news = user_news("all")
        status = 'Not Login'
        return render_template('./main/today_news.html', recommend_news = recommend_news, unrecommend_news = unrecommend_news, status = status)

@main_bp.route('/logout')
def logout():
    session['token'] = ''
    session['account'] = ''
    return redirect(f'/main')

# TODO
# @main_bp.route('/all_dates')
# def all_dates():
#     if 'token' in session:
#         all_news = item_data()
#         news_dates = all_news.sort_values('gattered_datetime').drop_duplicates(subset=['gattered_datetime'])
#         return render_template('./recommend/all_dates.html', news_date = news_dates)        
#     else:
#         return redirect(f'{BASE_URL}:8080/main')

# @main_bp.route('/all_news')
# def allnews():
#     if 'token' in session:
#         all_news = item_data()
#         date = request.args.get('gattered_datetime')
#         date_news = all_news.loc[all_news['gattered_datetime'] == date]
#         return render_template('./recommend/all_news.html', all_news = date_news)        
#     else:
#         return redirect(f'{BASE_URL}:8080/main')


@main_bp.route('/profile', methods = ['GET','POST'])
def profile():
    session['page'] = 'profile'
    if 'token' in session and session['token'] != '':
        is_login = 'True'
        line_state = os.urandom(16).hex()
        user = get_user(session['token'])
        if session['source'] == 'all':
            history = get_user_cliked(session['token'])
        else:
            history = history_data_source(session['token'], session['source'])
        if request.method == 'POST':
            data = request.get_json()
            source = data.get('source')
            session['source'] = source
        return render_template('./main/profile.html', user=user, history=history, is_login=is_login, line_state=line_state)
    else:
        is_login = 'False'
        return render_template('./main/profile.html', is_login=is_login, line_state=None)

@main_bp.route('/edit-profile', methods = ['GET','POST'])
def edit_profile():
    session['page'] = 'edit'
    is_login = 'False'
    if 'token' in session:
        is_login = 'True'
        user = get_user(session['token'])
        # current_app.logger.info(user)
        if session['source'] == 'all':
            history = get_user_cliked(session['token'])
        else:
            history = history_data_source(session['token'], session['source'])
    status = 'T'
    if request.method == 'POST':
        try:
            data = request.get_json()
            source = data.get('source')
            session['source'] = source
        except:
            account = request.form['account']
            password = request.form['password']
            email = request.form['email']
            line_id = request.form['line_id']
            
            if is_valid_email(email):
                status = 'True'
                update_user_data(session['token'], account, password, email, line_id)
            else:
                status = 'False'
    return render_template('./main/edit_profile.html', status = status, user = user, history = history, is_login = is_login)

@main_bp.route('/test', methods = ['GET'])
def test():
    return render_template('./layout/test.html')