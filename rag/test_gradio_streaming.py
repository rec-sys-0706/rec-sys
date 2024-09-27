from assistant import create_assistant
from openai import OpenAI
import gradio as gr

client = OpenAI()

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


def create_assistant(name, instructions, tools=None, vector_store_ids=None):
    _tools = []
    if tools is not None:
        if isinstance(tools, list):
            _tools.extend(tools)
        else:
            _tools.append(tools)
    
    if vector_store_ids is not None:
        _tools.append({"type": "file_search"})
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=_tools,
            tool_resources={"file_search": {"vector_store_ids": vector_store_ids}},
            model="gpt-4o-mini"
        )
    else:
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=_tools if _tools else None,
            model="gpt-4o-mini"
        )
    return assistant.to_dict()

def predict(message, history):
    history_openai_format = []
    for human, assistant_response in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant_response})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        assistant_id='asst_Bdr8pgk4kmkx8oU7ds6ZNPq0',
        messages=history_openai_format,
        temperature=1.0,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
    return partial_message

def gradio_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()  
        message = gr.Textbox(label="Your message")  
        state = gr.State([])  

        def respond(user_message, chat_history):
            chat_history = chat_history + [(user_message, "")]  
            response = predict(user_message, chat_history) 
            chat_history[-1] = (user_message, response)  
            return chat_history, chat_history 

        message.submit(respond, inputs=[message, state], outputs=[chatbot, state])  
    return demo

if __name__ == "__main__":
    gradio_interface().launch()
