from openai import OpenAI
import gradio as gr
import json

# 初始化 OpenAI 客戶端
client = OpenAI()

# 定義工具函數
def search_keyword(content: str):
    return f"{content} 是一種台灣美食，通常在中午和晚上供應。"

def no_this_function():
    return '不好意思，我沒有這個功能。'

# 工具集
toolkit = {
    'search_keyword': search_keyword,
    'no_this_function': no_this_function
}

# 定義工具規格
tools = [
    {
        "type": "function",
        "function": {
            "name": 'search_keyword',
            "description": "根據使用者輸入進行搜尋。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "使用者提供的搜尋內容。"
                    }
                },
                "required": ["content"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": 'no_this_function',
            "description": "當請求不符合任何已知功能時，回應此訊息。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

# 定義預測函式
def predict(message, history):
    # 將歷史紀錄轉換為 OpenAI 格式
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    # 發送請求到 OpenAI 並使用流式傳輸
    response = client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        messages=history_openai_format,
        temperature=1.0,
        tools=tools,
        tool_choice='auto',
        stream=True
    )

    partial_message = ""
    tool_called = False
    for chunk in response:
        # 檢查是否有工具調用
        if chunk.choices[0].finish_reason == 'tool_calls':
            tool_called = True
            tool_call = chunk.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # 根據被呼叫的工具名稱執行對應的函數
            for tool in tools:
                if function_name == tool['function']['name']:
                    args = [arguments.get(param) for param in tool['function']['parameters']['properties'].keys()]
                    function_result = toolkit[function_name](*args)

                    # 回傳工具的結果
                    yield function_result

        # 如果沒有調用工具，則逐步顯示模型的部分回應
        if chunk.choices[0].delta.content is not None and not tool_called:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

# 使用 Gradio 的 ChatInterface 並啟動
gr.ChatInterface(predict).launch()

