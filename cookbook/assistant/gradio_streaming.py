from openai import OpenAI
import gradio as gr

client = OpenAI()

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
  
    response = client.chat.completions.create(model='gpt-4o-mini-2024-07-18',
    messages= history_openai_format,
    temperature=1.0,
    stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message

gr.ChatInterface(predict).launch()


# from openai import OpenAI
# import gradio as gr
# from thread import create_thread, create_message, retrieve_thread_messages
# from classifier import ClassifierAgent
# client = OpenAI()
# agent = ClassifierAgent()

# threads = {}
# def predict(message, history, request: gr.Request):
#     if request.session_hash in threads:
#         thread = threads[request.session_hash]
#     else:
#         thread = create_thread('log.jsonl')
#         threads[request.session_hash] = thread
    
#     message = create_message(thread.id, 'user', content=message)
#     return agent.run(thread=thread)


# if __name__ == "__main__":
#     gr.ChatInterface(predict).launch()