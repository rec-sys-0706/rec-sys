from utils import save_jsonl
from openai import OpenAI
from openai.types.beta.thread import Thread
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.runs.run_step import RunStep
from typing import Literal
client = OpenAI()
def create_thread(saved_path=None) -> Thread:
    thread = client.beta.threads.create()
    if saved_path is not None:
        save_jsonl(saved_path, [thread.to_dict()], 'a')
    return thread
def retrieve_thread(thread_id, saved_path=None) -> Thread:
    thread = client.beta.threads.retrieve(thread_id)
    if saved_path is not None:
        save_jsonl(saved_path, [thread.to_dict()], 'a')
    return thread

def retrieve_thread_messages(thread_id, order: Literal['asc', 'desc']='desc', after=None) -> list[Message]:
    thread_messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        order=order,
        after=after
    )
    return thread_messages.data

def delete_thread(thread_id):
    response = client.beta.threads.delete(thread_id)
    return response

def create_message(thread_id: str, role: Literal['user', 'assistant'], content: str) -> Message:
    """Create message on thread."""
    thread_message = client.beta.threads.messages.create(
        thread_id,
        role=role,
        content=content,
    )
    return thread_message

def delete_message(thread_id, message_id):
    deleted_message = client.beta.threads.messages.delete(
        message_id=message_id,
        thread_id=thread_id,
    )
    print(deleted_message)

def retrieve_run(thread_id, run_id) -> Run:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )
    return run

def list_runs(thread_id) -> list[Run]:
    runs = client.beta.threads.runs.list(thread_id)
    return runs.data

def list_run_steps(thread_id, run_id) -> list[RunStep]:
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run_id
    )
    return run_steps.data


if __name__ == '__main__':
    # thread_id = 'thread_T9hbuXqOIH3PqNp6HBfO3lSN'
    # thread_id = 'thread_yiDAolCm5LT13Uyc67IXLLil'
    # thread_id = 'thread_vgQkjY54Jkmif9CJkP1oo2xy'
    # run_id = 'run_NAuZUaYcxGY2uhHYvBpH41N8'

    pass