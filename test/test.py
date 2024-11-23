from flask import Flask, request, abort, session, redirect, url_for
import logging
import os
import logging
from flask import Blueprint, request, session, redirect, url_for, abort
from dotenv import load_dotenv
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    PushMessageRequest
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import requests

linebot_bp = Blueprint('linebot', __name__)

# 環境變數設置
load_dotenv()
LINE_CHANNEL_ID = os.getenv("CHANNEL_ID")
LINE_CHANNEL_SECRET = os.getenv("LOGIN_CHANNEL_SECRET")
REDIRECT_URI = "https://recsys.csie.fju.edu.tw/api/callback/login"
configuration = Configuration(access_token=os.environ.get('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.environ.get('CHANNEL_SECRET'))

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 添加 session 密鑰


@app.route('/api/callback/login', methods=['GET'])
def callback_login():
    code = request.args.get('code')
    if not code:
        logging.warning('[LineBot] Authorization failed, no code received')
        return "授權失敗", 400

    # 獲取訪問令牌
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": LINE_CHANNEL_ID,
        "client_secret": LINE_CHANNEL_SECRET,
    }

    token_response = requests.post("https://api.line.me/oauth2/v2.1/token", data=token_data)
    # 檢查是否成功取得令牌
    if token_response.status_code != 200:
        logging.error('[LineBot] Failed to get access token')
        return "獲取訪問令牌失敗", 500

    access_token = token_response.json().get("access_token")
    if not access_token:
        logging.error('[LineBot] Access token not found in response')
        return "未找到訪問令牌", 500

    # 使用令牌取得用戶資料
    headers = {"Authorization": f"Bearer {access_token}"}
    profile_response = requests.get("https://api.line.me/v2/profile", headers=headers)
    if profile_response.status_code != 200:
        logging.error('[LineBot] Failed to get user profile data')
        return "獲取個人資料失敗", 500

    user_data = profile_response.json()
    user_id = user_data.get("userId")
    if not user_id:
        logging.error('[LineBot] User ID not found in profile data')
        return "未找到用戶ID", 500

    # 儲存用戶資料的邏輯在這裡
    logging.info(f'[LineBot] User profile data retrieved: {user_data}')

    # 推送訊息給用戶
    messaging_api = MessagingApi(ApiClient(configuration))
    message = TextMessage(text="歡迎使用我們的服務！")
    push_request = PushMessageRequest(to=user_id, messages=[message])
    messaging_api.push_message(push_request)

    return redirect('/test')

@app.route('/test', methods=['GET'])
def test():
    return '成功返回測試驗面'

@app.route('/api/callback', methods=['POST']) # /callback
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    logging.info(f'[LineBot] receive webhook request, signature: {signature}, content: {body}')

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logging.info("invalid signature. please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

app.run(host='0.0.0.0', port=5000, debug=True)