import os
import logging

from dotenv import load_dotenv
from flask import Blueprint, request, abort, jsonify

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
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

load_dotenv()

linebot_bp = Blueprint('callback', __name__)

configuration = Configuration(access_token=os.environ.get('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.environ.get('CHANNEL_SECRET'))

def chatbot():
    # 定義您的 chatbot function
    return "這是來自 chatbot 函數的回應。"

@linebot_bp.route('', methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logging.info("無效的簽名。請檢查您的頻道存取權杖和頻道密鑰。")
        abort(400)

    return 'OK'

@handler.route('/broadcast', methods=['POST'])
def broadcast():
    data = request.json
    message = data.get("message")
    # 處理廣播邏輯
    print("收到廣播訊息:", message)
    return "廣播已成功發送", 200

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    logging.info(f'[LineBot] {user_message}')

    # 檢查使用者訊息中的特定指令
    if 'Hello chatbot' in user_message:
        # 呼叫 chatbot 函數並將回應發送回去
        response = chatbot()
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=response)]
                )
            )
        return  # 在 chatbot() 回應後結束函數

    # 如果沒有找到特定指令，則預設回應
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=user_message)]
            )
        )