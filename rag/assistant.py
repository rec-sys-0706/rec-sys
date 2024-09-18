from openai import OpenAI
client = OpenAI()

def list_assistants(order='desc', limit='20') -> None:
    assistants = client.beta.assistants.list(
        order=order,
        limit=limit,
    )
    for asst in assistants:
        print(f'{asst.id}, {asst.name}')

def create_assistant(name, instructions, tools=None, vector_store_ids=None):
    _tools = []
    if tools is not None:
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
            model="gpt-4o-mini"
        )
    return assistant.to_dict()

if __name__ == '__main__':
    # create_assistant('Summarizer', """
    #     You are expert in analyzing and answering questions about any story, including fiction and non-fiction. You provide thoughtful, accurate answers based on the given content. You should focus on understanding key elements of the story, such as plot, characters, themes, settings, and the underlying message. You should give clear, concise answers to direct questions while providing detailed explanations when necessary. You should avoid adding details not present in the original story and refrain from interpreting beyond what is supported by the text. When clarification is needed, you should ask the user for more information or specific details to ensure accurate responses.
    #     """)


    print(assistant)