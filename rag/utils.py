import json
from typing import Literal
import tiktoken
enc = tiktoken.get_encoding("o200k_base") # gpt-4o

def get_tokens(text: str):
    return enc.encode(text)
    
def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def save_jsonl(filepath, data: list[dict], mode: Literal['w', 'a']='a'):
    with open(filepath, mode) as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
