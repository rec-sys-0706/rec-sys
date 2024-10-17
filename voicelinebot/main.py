import os
import whisper
import tempfile
import jieba.analyse
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage, AudioMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, MessageAction, URIAction, CarouselTemplate, CarouselColumn

app = Flask(__name__)

# Set Environment Variables
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# load model
model = whisper.load_model("small")

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")
    app.logger.info(f"Signature: {signature}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=AudioMessage)
def audio_message(event):
    content = line_bot_api.get_message_content(event.message.id)
    audio_content = content
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        for chunk in audio_content.iter_content():
            tf.write(chunk)
        tf.seek(0)  # 文件從開頭讀取
        result = model.transcribe(tf.name, initial_prompt="這部電影如同一場夢幻的旅程,我給它的評分是10分。") 
        jieba.load_userdict('keywords.txt')
        keyword = jieba.analyse.extract_tags(result["text"], topK=5, withWeight=False)
        
        if '電影' in keyword:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text='以下是搜尋到的結果'))
        elif '評分' in keyword:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text='電影評分為'))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result["text"]))
        
        # 如果关键词中包含"新聞"，发送按钮模板消息
        if '新聞' in keyword:
            line_bot_api.push_message(event.source.user_id, TemplateSendMessage(
            alt_text='CarouselTemplate',
            template=CarouselTemplate(
                columns=[
                    CarouselColumn(
                        thumbnail_image_url='https://thumb.photo-ac.com/74/741d67cb414d74db410ef13a350a69ae_t.jpeg',
                        title='News Recommend',
                        text='推薦新聞網',
                        actions=[
                            URIAction(
                                    label='Recommand News',
                                    uri='http://127.0.0.1:8080/main/'
                                )
                        ]
                    )
                ]
            )
        )
    )

if __name__ == "__main__":
    from waitress import serve
    print("Starting server on http://0.0.0.0:8000")
    serve(app, host='0.0.0.0', port=8000)
