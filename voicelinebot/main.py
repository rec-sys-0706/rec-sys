import os
import whisper
import tempfile
import jieba.analyse
from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage, AudioMessage, TemplateSendMessage, CarouselTemplate, CarouselColumn, URIAction
import logging  # 引入日誌模組

app = Flask(__name__)

# 設定日誌
logging.basicConfig(level=logging.INFO)  # 設定日誌級別

# 取得 LINE Bot API 資訊並進行檢查
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
channel_secret = os.getenv("LINE_CHANNEL_SECRET")

if not channel_access_token or not channel_secret:
    raise EnvironmentError("請檢查環境變數設定：LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET")

# 初始化 LINE API
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# 加載 Whisper 模型
model = whisper.load_model("small")

# 定義 chatbot() 函數
def chatbot():
    return "呼叫 chatbot() 成功！這是 AI 對話的回應。"

# 定義廣播訊息函數
def broadcast_message(message_text):
    line_bot_api.broadcast(TextSendMessage(text=message_text))

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info(f"Received request body: {body}")  # 打印請求的 body
    
    try:
        handler.handle(body, signature)
        app.logger.info("Webhook event handled successfully.")  # 處理成功的日誌
    except InvalidSignatureError:
        app.logger.error("無效的簽章，請檢查 Channel Access Token 或 Channel Secret 設定。")
        abort(400)
    except Exception as e:
        app.logger.error(f"Error handling webhook event: {e}")
        abort(500)
    
    return "OK"

@app.route("/broadcast", methods=["POST"])
def broadcast():
    data = request.get_json()
    message_text = data.get("message", "這是預設的廣播訊息")
    
    try:
        broadcast_message(message_text)
        return jsonify({"status": "success", "message": "廣播訊息已發送"})
    except Exception as e:
        app.logger.error(f"廣播訊息失敗：{e}")
        return jsonify({"status": "error", "message": "廣播訊息失敗"}), 500

# 音訊訊息事件處理
@handler.add(MessageEvent, message=AudioMessage)
def audio_message(event):
    content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        for chunk in content.iter_content():
            tf.write(chunk)
        
        result = model.transcribe(tf.name, initial_prompt="這部電影如同一場夢幻的旅程,我給它的評分是10分。")
        
        # 嘗試載入自定義關鍵詞
        try:
            jieba.load_userdict("keywords.txt")
        except Exception as e:
            app.logger.warning(f"無法載入 keywords.txt：{e}")
        
        keyword = jieba.analyse.extract_tags(result["text"], topK=5, withWeight=False)
        
        # 偵測關鍵字
        if "機器人" in result["text"]:
            chatbot_response = chatbot()
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=chatbot_response))
            return
        
        # 根據關鍵字進行回應
        if "電影" in keyword:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="以下是搜尋到的結果"))
        elif "評分" in keyword:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="電影評分為"))
        elif "新聞" in keyword:
            # 發送 CarouselTemplate 推薦新聞
            line_bot_api.push_message(event.source.user_id, TemplateSendMessage(
                alt_text="CarouselTemplate",
                template=CarouselTemplate(
                    columns=[
                        CarouselColumn(
                            thumbnail_image_url="https://thumb.photo-ac.com/74/741d67cb414d74db410ef13a350a69ae_t.jpeg",
                            title="News Recommend",
                            text="推薦新聞網",
                            actions=[URIAction(label="Recommand News", uri="http://127.0.0.1:8080/main/")]
                        )
                    ]
                )
            ))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result["text"]))

if __name__ == "__main__":
    from waitress import serve
    app.logger.info("Starting server on http://0.0.0.0:8000")  # 添加日誌
    serve(app, host="0.0.0.0", port=8000)
