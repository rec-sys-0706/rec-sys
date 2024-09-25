from openai import OpenAI
client = OpenAI()

def list_assistants(order='desc', limit='20') -> None:
    assistants = client.beta.assistants.list(
        order=order,
        limit=limit,
    )
    for asst in assistants:
        print(f'{asst.id}, {asst.name}')

# def create_assistant(name, instructions, tools=None, vector_store_ids=None):
#     _tools = []
#     if tools is not None:
#         _tools.append(tools)
#     if vector_store_ids is not None:
#         _tools.append({"type": "file_search"})
#         assistant = client.beta.assistants.create(
#             name=name,
#             instructions=instructions,
#             tools=_tools,
#             tool_resources={"file_search": {"vector_store_ids": vector_store_ids}},
#             model="gpt-4o-mini"
#         )
#     else:
#         assistant = client.beta.assistants.create(
#             name=name,
#             instructions=instructions,
#             model="gpt-4o-mini"
#         )
#     return assistant.to_dict()

def create_assistant(name, instructions, tools=None, vector_store_ids=None):
    _tools = []
    if tools is not None:
        _tools.extend(tools)  # 使用 extend 而不是 append
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
            tools=_tools,
            model="gpt-4o-mini"
        )
    return assistant.to_dict()

if __name__ == '__main__':
    # create_assistant('Summarizer', """
    #     You are expert in analyzing and answering questions about any story, including fiction and non-fiction. You provide thoughtful, accurate answers based on the given content. You should focus on understanding key elements of the story, such as plot, characters, themes, settings, and the underlying message. You should give clear, concise answers to direct questions while providing detailed explanations when necessary. You should avoid adding details not present in the original story and refrain from interpreting beyond what is supported by the text. When clarification is needed, you should ask the user for more information or specific details to ensure accurate responses.
    #     """)
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
    
    create_assistant("Intent_Recognizer",
                     """You are message handler,and your task is to analyze the user's input and recognize their intent.
                        Perform intent recognition based on user requests.  
                        Ensure your reasoning is accurate before finalizing your output. Keep the tone direct and focused. 
                        """,
                    tools=tools)

    


    #print(assistant)