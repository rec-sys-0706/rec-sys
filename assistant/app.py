import os
import openai
import gradio as gr

from functools import partial
from dotenv import load_dotenv

def initialize():
    initialMessage = [
        {
            "role": "system",
            "content": """你是一個親切的天氣助手，將會協助我取得我需要的天氣資訊。
            我將透過以下形式來提供你查詢的資訊：
            城市：日期
            你需要回答我這個城市在我指定的日期將會是什麽樣的天氣。
            """
        },
        {
            "role": "user",
            "content": """請提供我以下城市的天氣資訊
            台北：2023/2/28
            """,
        },
        {
            "role": "assistant",
            "content": "陰雨"
        }
    ]
    return initialMessage

def updateMessageList(message, role, messageList):
    try:
        messageList.append({
            "role": role,
            "content": message,
        })
    except Exception as e:
        print(e)
    
    return messageList

def getResponse(promot, messageList):
    # 將使用者輸入內容更新至訊息紀錄
    updateMessageList(promot, 'user', messageList)

    # 與API互動並取得回應
    responseDict = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messageList,
    )
  
    # 擷取回覆字串
    responseMessage = responseDict['choices'][0]['message']['content']

    # 將回覆更新至訊息紀錄
    updateMessageList(responseMessage, 'assistant', messageList)
    
    userContext = [content['content'] for content in messageList if content['role'] == 'user']
    assistantContext = [content['content'] for content in messageList if content['role'] == 'assistant']
    
    # 構建用戶對話記錄
    response = [(_user, _response) for _user, _response in zip(userContext[1:], assistantContext[1:])]

    return response, messageList

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    messageList = initialize()

    # 使用 partial 函數將 messageList 作為一個固定的參數
    partialGetResponse = partial(getResponse, messageList=messageList)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        state = gr.State([])

        with gr.Row():
            text = gr.Textbox(
                show_label=False,
                placeholder="對ChatGPT說些什麽.....",
            )
        
        # 注意 partialGetResponse 的參數設置，將返回值對應 chatbot 和 state
        text.submit(partialGetResponse, [text], [chatbot, state])

    demo.launch()

if __name__ == '__main__':
    main()

