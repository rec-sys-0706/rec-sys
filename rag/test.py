from assistant import create_assistant

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
                tools=tools
                )