import pickle
from openai import OpenAI
import json
client = OpenAI()

# ! Tools
secrets = ['ppqq454', 'bbcc102', 'asdasd44']
def get_secret(id: str):
    return f"Secret is {secrets[int(id)]}."

toolkit = {
    get_secret.__name__: get_secret
}

tools = [
    {
        "type": "function",
        "function": {
            "name": f"{get_secret.__name__}",
            "description": "Get the secret",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The secret's ID."
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False
            },
        }
    }
]


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the secret 2?"}
]

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice='auto'
)

choice = completion.choices[0]
messages.append(choice.message)
if choice.finish_reason == 'tool_calls':
    tool_call = choice.message.tool_calls[0]
    print(tool_call)
    function_name = tool_call.function.name
    print(function_name)
    arguments = json.loads(tool_call.function.arguments)

    for tool in tools:
        if function_name == tool['function']['name']:
            args = [arguments.get(param) for param in tools[0]['function']['parameters']['properties'].keys()]
            function_result = toolkit[function_name](*args)
            tool_call_result_message = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": choice.message.tool_calls[0].id
            }
            messages.append(tool_call_result_message)


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice='auto'
)

with open('./temp.pkl', 'wb') as fout:
    pickle.dump(completion.choices, fout)
