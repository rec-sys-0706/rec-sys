import os
from datetime import datetime, timedelta
import logging

from flask import Blueprint, render_template, request, session, redirect, current_app
from .utils import register, login, get_user, update_user_data, msg
from .utils import get_history
from .utils import get_document, get_document_for_user
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
    # TODO 登入有給uuid，為什麼還要從session拿？
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
            session['account'] = account
            session['page'] = 'index'
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


@main_bp.route('/about', methods = ['GET','POST'])
def about():
    session['page'] = 'about'
    try:
        data = request.get_json()
        source = data.get('source')
        session['source'] = source
    except:
        print('error')
    return render_template('./main/about.html')

@main_bp.route('/', methods = ['GET'])
def index():
    # initialize session
    if session.get('source', None) is None:
        session['source'] = 'news'
    
    # Session
    source = session.get('source', 'news')
    # Query
    is_recommend = request.args.get('is_recommend', session.get('is_recommend', 'False'))
    date = request.args.get('date', session.get('date', datetime.today().strftime('%Y-%m-%d')))
    # Update session
    session['is_recommend'] = is_recommend
    session['date'] = date
    # Process date
    _date = datetime.strptime(date, '%Y-%m-%d')
    _prev_date = _date - timedelta(days=1)
    _next_date = _date + timedelta(days=1)

    prev_date = _prev_date.strftime('%Y-%m-%d') if _prev_date > datetime(2023, 12, 31) else None
    next_date = _next_date.strftime('%Y-%m-%d') if _next_date < datetime.today() else None

    news = []
    if session.get('token', None) is not None:
        news = get_document_for_user(session['token'], source, is_recommend, date)
    elif is_recommend == 'False':
        news = get_document(source, date)


    session_str = '\n'.join([f'{k}: {v}' for k, v in session.items()])

    logging.info((
        f'session: \n{session_str}\n'
        f'is_recommend: {is_recommend}\n'
        f'date: {date}\n'
        f'source: {source}\n'
        f'news length: {len(news)}\n'
    ))
    return render_template('./main/show_news.html', news=news, is_recommend=is_recommend, date=date, prev_date=prev_date, next_date=next_date)



# @main_bp.route('/click', methods = ['POST'])
# def click():
#     if 'token' in session and session['token'] != '':
#         if request.method == 'POST':
#             try:                    
#                 data = request.get_json()
#                 link = data.get('link')
#                 current_app.logger.info(session)
#                 print(data)
#                 if session['source'] == 'all':
#                     click_data(session['token'], link)
#                 else:
#                     click_data_source(session['token'], link, session['source'])
#             except:
#                 data = request.get_json()
#                 source = data.get('source')
#                 session['source'] = source

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

    user = None
    history = None
    line_state = None

    if session.get('token', None) is not None:
        user = get_user(session['token'])
        history = get_history(session['token'], session['source'])
        line_state = os.urandom(16).hex()

        if request.method == 'POST':
            data = request.get_json()
            source = data.get('source')
            session['source'] = source

    return render_template('./main/profile.html', user=user, history=history, line_state=line_state)

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