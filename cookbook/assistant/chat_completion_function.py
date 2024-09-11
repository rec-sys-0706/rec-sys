from openai import OpenAI
client = OpenAI()
tools = [
{
    "type": "function",
    "function": {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
    }
}
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(completion.to_json())


    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": 'insert_row',
    #             "description": "Insert a paper into dataframe.",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "name": {
    #                         "type": "string",
    #                         "description": "user name"
    #                     },
    #                     "keywords": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "string"
    #                         },
    #                         "description": ""
    #                     },
    #                 },
    #                 "required": [
    #                     "research_area",
    #                     "task",
    #                     "contribution_type",
    #                     "model_type",
    #                     "dataset",
    #                     "keywords"
    #                 ],
    #                 "additionalProperties": False,
    #                 "strict": True,
    #             },
    #             "strict": True
    #         }
    #     }
    # ]