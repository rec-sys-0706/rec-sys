import gradio as gr
from openai import OpenAI
import json

client = OpenAI()

#定義函數search_keyword()以及no_this_function()
def search_keyword(content: str):
    return f"{content}是一種台灣美食，通常在中午和晚上供應"

def no_this_function():
    return 'success:true'

#將函數放入字典，供後面程式碼使用
toolkit = {
    'search_keyword': search_keyword,
    'no_this_function': no_this_function
}

#定義兩種可用工具，包含類型、名稱及參數
tools = [
    {
        "type": "function",
        "function": {
            "name": 'search_keyword',
            "description": "Perform a search based on user input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The search content provided by the user."
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
            "description": "Responds with '不好意思，我沒有這個功能' if the request does not match any known function.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

#設定最初始的對話
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}
]

#定義聊天函數，處理user_input
def chat(user_input):
    global messages 
    messages.append({'role': 'user', 'content': user_input})

    #請求完成對話
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice='auto'
    )
    
    choice = completion.choices[0]
    messages.append(choice.message)

    #檢查是否有呼叫工具
    if choice.finish_reason == 'tool_calls':
        tool_call = choice.message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        #尋找呼叫名稱匹配的工具
        for tool in tools:
            if function_name == tool['function']['name']:
                args = [arguments.get(param) for param in tool['function']['parameters']['properties'].keys()]
                function_result = toolkit[function_name](*args) #調用對應的函數
                tool_call_result_message = {
                    "role": "tool",
                    "content": function_result,
                    "tool_call_id": tool_call.id
                }
                messages.append(tool_call_result_message)

        #再次請求完成對話
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice='auto'
        )

        choice = completion.choices[0]
        messages.append(choice.message)

    output = []
    for message in messages:
        #檢查訊息的類型
        if isinstance(message, dict):  
            role = message['role']
            content = message['content']
        else:
            role = message.role
            content = message.content
        
        if role in ['user', 'assistant'] and content is not None:
            output.append(f"{role}: {content}")

    return "\n".join(output)  

#創建gradio網頁介面
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Chatbot", description="與聊天助手互動")
iface.launch()

