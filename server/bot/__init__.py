import os
import logging
import schedule
import time
from threading import Thread

from flask import Blueprint, request, abort, redirect
import requests

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    PushMessageRequest
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

linebot_bp = Blueprint('callback', __name__)

BASE_URL = os.getenv("BASE_URL")
REDIRECT_URI = "https://recsys.csie.fju.edu.tw/api/callback/login"
LINE_CHANNEL_ID = os.getenv("CHANNEL_ID")
LINE_CHANNEL_SECRET = os.getenv("LOGIN_CHANNEL_SECRET")
configuration = Configuration(access_token=os.environ.get('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.environ.get('CHANNEL_SECRET'))

@linebot_bp.route('', methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logging.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# login
@linebot_bp.route('/login', methods=['GET'])
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

    logging.info(f'[LineBot] User profile data retrieved: {user_data}')

    # 推送訊息給用戶
    messaging_api = MessagingApi(ApiClient(configuration))
    message = TextMessage(text=(
        f"歡迎 {user_data.get('displayName')} 使用我們的服務！\n"
        f"您已成功註冊會員\n"
        f"使用 Line 登入網站即可閱讀文章！\n"
    ))
    push_request = PushMessageRequest(to=user_id, messages=[message])
    # messaging_api.push_message(push_request)
    return redirect('/main/profile')

# 獲取用戶資料和多條消息
def fetch_user_messages():
    """
    從資料庫 API 獲取需要發送的用戶清單及多條消息
    返回格式範例：
    [
        {"userId": "U1234567890", "messages": ["Message 1", "Message 2", "Message 3", ...]},
        {"userId": "U0987654321", "messages": ["Message 1", "Message 2", "Message 3", ...]}
    ]
    """
    try:
        # 資料庫 API 用來拉取推薦的文章和用戶消息
        db_api_url = f"{os.environ.get('ROOT')}:5000/api/callback/recommended_articles"
        response = requests.get(db_api_url)
        if response.status_code != 200:
            logging.error("[LineBot] Failed to fetch user messages")
            return []

        return response.json()  # 假設 API 返回 JSON 格式的數據
    except Exception as e:
        logging.error(f"[LineBot] Error fetching user messages: {e}")
        return []

def push_scheduled_messages():
    """
    向每位用戶發送多條消息
    """
    user_messages = fetch_user_messages()
    if not user_messages:
        logging.info("[LineBot] No messages to send")
        return

    with ApiClient(configuration) as api_client:
        messaging_api = MessagingApi(api_client)
        for entry in user_messages:
            user_id = entry.get("userId")
            messages = entry.get("messages", [])
            if not user_id or not messages:
                logging.warning("[LineBot] Skipping invalid user/messages entry")
                continue

            try:
                # 創建多條訊息
                message_objects = [TextMessage(text=msg) for msg in messages]
                push_request = PushMessageRequest(to=user_id, messages=message_objects)
                messaging_api.push_message(push_request)
                logging.info(f"[LineBot] Messages sent to {user_id}: {messages}")
            except Exception as e:
                logging.error(f"[LineBot] Failed to send messages to {user_id}: {e}")

# 設置定時任務
schedule.every().day.at("12:00").do(push_scheduled_messages)

def run_scheduler():
    logging.info("[LineBot] Scheduler started")
    while True:
        schedule.run_pending()
        time.sleep(1)

@linebot_bp.route('/start_scheduler', methods=['POST'])
def start_scheduler():
    """
    啟動 Scheduler 的 API，適用於 Flask 部署
    """
    thread = Thread(target=run_scheduler)
    thread.daemon = True
    thread.start()
    return "Scheduler started", 200