from flask import Blueprint, render_template, request, session, redirect, url_for
import requests
import os
import re
from config import register, item_data, login, BASE_URL, click_data, user_data, update_user_data, msg, get_recommend, get_unrecommend, get_user_cliked

main_bp = Blueprint('main', 
                    __name__, 
                    template_folder='templates',
                    static_folder='static', 
                    url_prefix='/main')

# LINE API配置
LINE_CHANNEL_ID = "CANNEL_ID"
LINE_CHANNEL_SECRET = "LOGIN_CHANNEL_SECRET"
REDIRECT_URI = f"REDIRECT_URI"

@main_bp.route('/', methods = ['GET', 'POST'])
def index():
    status = 'T'
    if request.method == 'POST':
        account = request.form['account']
        password = request.form['password']
        msg = login(account, password)
        if msg is None:
            status = 'F'
        else:
            session['token'] = msg
            return render_template('./recommend/about.html')
    return render_template('./main/login.html', status=status)

def is_valid_email(email):
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    return re.fullmatch(pattern, email) is not None

@main_bp.route('/signup', methods=['GET', 'POST'])
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
            if message == 'exists':
                status = 'F'
        else:
            status = 'False' 
    return render_template('./main/signup.html', status=status)

# Line Login
@main_bp.route('/login')
def login_with_line():
    state = os.urandom(16).hex()
    session['state'] = state
    login_url = (
        f"https://access.line.me/oauth2/v2.1/authorize?response_type=code"
        f"&client_id={LINE_CHANNEL_ID}&redirect_uri={REDIRECT_URI}"
        f"&state={state}&scope=profile%20openid%20email"
    )
    return redirect(login_url)

@main_bp.route('/callback')
def callback():
    if request.args.get('state') != session.get('state'):
        return "State mismatch", 400

    code = request.args.get('code')
    if not code:
        return "Authorization failed", 400

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": LINE_CHANNEL_ID,
        "client_secret": LINE_CHANNEL_SECRET,
    }
    token_response = requests.post("https://api.line.me/oauth2/v2.1/token", data=token_data)
    access_token = token_response.json().get("access_token")

    headers = {"Authorization": f"Bearer {access_token}"}
    profile_response = requests.get("https://api.line.me/v2/profile", headers=headers)
    user_data = profile_response.json()

    # TODO: 儲存用戶資料到資料庫
    save_user(user_data)
    session['user'] = user_data

    return redirect(url_for('main.recommend'))  # Redirect to the main recommend page after login

def save_user(user_data):
    # TODO: 實作儲存用戶資料到資料庫的邏輯
    pass

@main_bp.route('/recommend')
def recommend():
    return render_template('./recommend/about.html')

@main_bp.route('/today_news', methods=['GET', 'POST'])
def today_news():
    if 'token' in session:
        if request.method == 'POST':
            data = request.get_json()
            link = data.get('link')
            click_data(session['token'], link)
        recommend_new = get_recommend(session['token'])
        unrecommend_news = get_unrecommend(session['token'])
        return render_template('./recommend/today_news.html', recommend_new=recommend_new, unrecommend_news=unrecommend_news)
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/all_dates')
def all_dates():
    if 'token' in session:
        all_news = item_data()
        news_dates = all_news.sort_values('gattered_datetime').drop_duplicates(subset=['gattered_datetime'])
        return render_template('./recommend/all_dates.html', news_date=news_dates)        
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/all_news')
def allnews():
    if 'token' in session:
        all_news = item_data()
        date = request.args.get('gattered_datetime')
        date_news = all_news.loc[all_news['gattered_datetime'] == date]
        return render_template('./recommend/all_news.html', all_news=date_news)        
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/profile')
def profile():
    if 'token' in session:
        user = user_data(session['token'])
        history = get_user_cliked(session['token'])
        return render_template('./recommend/profile.html', user_data=user, history=history)
    else:
        return redirect(f'{BASE_URL}:8080/main')

@main_bp.route('/revise', methods=['GET', 'POST'])
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
        return render_template('./recommend/revise.html', status=status, user_data=user, history=history)

@main_bp.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('./news.html', title=title, content=content)