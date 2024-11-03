import requests
import pandas as pd
import hmac
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import jwt
from werkzeug.security import generate_password_hash

load_dotenv()
ROOT = os.environ.get('ROOT')
BASE_URL = os.environ.get('BASE_URL')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')


def get_signature(payload=''):
    # Get SQL_SECRET
    #secret_key = '123'
    secret_key = os.environ.get('SQL_SECRET')
    # Compute the HMAC-SHA256 signature
    hash_object = hmac.new(secret_key.encode('utf-8'), msg = payload.encode('utf-8'), digestmod=hashlib.sha256)
    signature = "sha256=" + hash_object.hexdigest()
    return signature
# payload = '{"example": "data"}' # 如果是 GET 則不用payload
# Prepare the headers, including the x-hub-signature-256

def format_date(date_str):
    return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z").strftime("%b %d, %Y")

#註冊
def register(email, account, password):
    data = {
        "account" : f"{account}",
        "password" : f"{password}",
        "email" : f"{email}",
        "line_id" : ""
    }
    response = requests.post(f'{ROOT}/api/user/register', json = data)
    return response.content

#登入
def login(account, password):
    data = {
        "account" : f"{account}",
        "password" : f"{password}"
    }
    response = requests.post(f'{ROOT}/api/user/login', json = data)
    response_json = json.loads(response.content)
    access_token = response_json.get('access_token')
    return access_token

#解碼
def access_decode(access_token):   
    text = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=['HS256'])
    id = text.get('sub')
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    response = requests.get(f'{ROOT}/api/user/{id}', headers=headers)
    return response.content

#修改user
def update_user_data(access_token, account, password, email, line_id):
    text = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=['HS256'])
    id = text.get('sub')
    headers = {
        "Authorization" : f'Bearer {access_token}'
    }
    password = generate_password_hash(password)
    user_data = {
        "account" : f"{account}",
        "password" : f"{password}",
        "email" : f"{email}",
        "line_id" : f"{line_id}"
    }
    requests.put(f"{ROOT}/api/user/{id}", headers=headers, json=user_data)

def msg(text):
    string_data = text.decode('utf-8')
    data_dict = json.loads(string_data)
    message = data_dict.get('msg')
    if message == "Username already exists":
        message = 'exists'
    return message

#獲得user
def user_data(access_token):
    texts = access_decode(access_token)
    decoded_text = texts.decode('utf-8')
    json_data = json.loads(decoded_text)
    data = json_data['data']
    user_data = pd.DataFrame([data])
    return user_data

def get_recommend(access_token):
    user = user_data(access_token)
    id = user['uuid'].iloc[0]
    response = requests.get(f"{ROOT}/api/user_history/recommend/{id}")
    data = response.json()
    items = []
    try:
        for entry in data:
            item_data = entry['item']
            item_data['recommendation_log_uuid'] = entry['recommendation_log_uuid']
            items.append(item_data)
        item = pd.DataFrame(items)
        #item = item.sort_values('title')
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = ''
    return item

def get_unrecommend(access_token):
    user = user_data(access_token)
    id = user['uuid'].iloc[0]
    response = requests.get(f"{ROOT}/api/user_history/unrecommend/{id}")
    #data = response.json()
    data = [{'item': {'abstract': 'Alcon Entertainment says it denied a request to use material from the film at the Tesla cybercab event.', 'data_source': 'bbc_news', 'gattered_datetime': 'Tue, 22 Oct 2024 19:17:06 GMT', 'link': 'https://www.bbc.com/news/articles/ce3z37dpvl9o', 'title': 'Blade Runner 2049 maker sues Musk over robotaxi images', 'uuid': '0fa5b2cc-5e22-42ee-8aea-032ebaa54832'}, 'recommendation_log_uuid': '3cb0a381-e69b-4aa3-8bd6-4b84fa0afcb8'}, {'item': {'abstract': 'It is safe, could speed up diagnosis and relieve NHS pressure, the health assessment body says.', 'data_source': 'bbc_news', 'gattered_datetime': 'Tue, 22 Oct 2024 19:16:40 GMT', 'link': 'https://www.bbc.com/news/articles/c2060gy9zy1o', 'title': 'AI to help doctors spot broken bones on X-rays', 'uuid': '0e57ccac-a75e-4c70-9854-24bf2cb5d794'}, 'recommendation_log_uuid': '3c16731f-05ec-44f6-ab51-a4e9eeb763cd'}, {'item': {'abstract': 'Artificial intelligence is being used to generate paintings, images and even sculptures, with some selling for thousands of dollars. Do we need to reframe our definition of art?', 'data_source': 'bbc_news', 'gattered_datetime': 'Mon, 21 Oct 2024 19:16:59 GMT', 'link': 'https://www.bbc.com/future/article/20241018-ai-art-the-end-of-creativity-or-a-new-movement', 'title': 'The AI art redefining creativity', 'uuid': 'b901d012-e197-4da4-bbb0-7320fc2e22ac'}, 'recommendation_log_uuid': 'a7dd83e7-87fc-41f1-b13a-43dca5ff01a8'}, {'item': {'abstract': 'The broadcaster has posted a job ad for someone to use AI to "shape the future of content creation".', 'data_source': 'bbc_news', 'gattered_datetime': 'Mon, 21 Oct 2024 19:16:48 GMT', 'link': 'https://www.bbc.com/news/articles/c62m24r7r85o', 'title': 'Derry Girls creator hits out at ITV over AI plans', 'uuid': '2321d36f-f68e-4752-8a83-153ed004c80d'}, 'recommendation_log_uuid': '350a76bf-a53d-4ecf-9015-dd8a170219a8'}, {'item': {'abstract': 'Chinese technology giant ByteDance denied reports that the incident caused more than $10m of damage.', 'data_source': 'bbc_news', 'gattered_datetime': 'Mon, 21 Oct 2024 19:16:28 GMT', 'link': 'https://www.bbc.com/news/articles/c7v62gg49zro', 'title': 'TikTok owner sacks intern for sabotaging AI project', 'uuid': '489b0c23-10db-4b97-b2b0-1ce9923671f1'}, 'recommendation_log_uuid': 'a55d7e79-a0be-4977-a1fc-ddeab93e2aa7'}, {'item': {'abstract': 'AI was the big theme at Gitex Global, held last week at Dubai’s World Trade Centre.', 'data_source': 'cnn_news', 'gattered_datetime': 'Mon, 21 Oct 2024 00:00:00 GMT', 'link': 'https://www.cnn.com/2024/10/21/middleeast/ai-robots-gitex-dubai-spc/index.html', 'title': 'AI and robots take center stage at ‘world’s largest tech event’', 'uuid': '15309bc9-b62a-446a-9b27-0ba4f929c3d3'}, 'recommendation_log_uuid': '8239604b-12a8-44f4-a326-14f7015cbf8c'}, {'item': {'abstract': 'Reward models are critical in techniques like Reinforcement Learning from Human Feedback (RLHF) and Inference Scaling Laws, where they guide language model alignment and select optimal responses. Despite their importance, existing reward model benchmarks often evaluate models by asking them to distinguish between responses generated by models of varying power. However, this approach fails to assess reward models on subtle but critical content changes and variations in style, resulting in a low correlation with policy model performance. To this end, we introduce RM-Bench, a novel benchmark designed to evaluate reward models based on their sensitivity to subtle content differences and resistance to style biases. Extensive experiments demonstrate that RM-Bench strongly correlates with policy model performance, making it a reliable reference for selecting reward models to align language models effectively. We evaluate nearly 40 reward models on RM-Bench. Our results reveal that even state-of-the-art models achieve an average performance of only 46.6%, which falls short of random-level accuracy (50%) when faced with style bias interference. These findings highlight the significant room for improvement in current reward models. Related code and data are available at this https URL.', 'data_source': 'hf_paper', 'gattered_datetime': 'Mon, 21 Oct 2024 00:00:00 GMT', 'link': 'https://huggingface.co/papers/2410.16184', 'title': 'RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style', 'uuid': '78da1bd5-54bd-4a96-954b-36c2f417dc61'}, 'recommendation_log_uuid': 'b8b4912a-fd1e-4743-a95e-aba56fd66047'}, {'item': {'abstract': 'Knowledge distillation (KD) aims to transfer knowledge from a large teacher model to a smaller student model. Previous work applying KD in the field of large language models (LLMs) typically focused on the post-training phase, where the student LLM learns directly from instructions and corresponding responses generated by the teacher model. In this paper, we extend KD to the pre-training phase of LLMs, named pre-training distillation (PD). We first conduct a preliminary experiment using GLM-4-9B as the teacher LLM to distill a 1.9B parameter student LLM, validating the effectiveness of PD. Considering the key impact factors of distillation, we systematically explore the design space of pre-training distillation across four aspects: logits processing, loss selection, scaling law, and offline or online logits. We conduct extensive experiments to explore the design space of pre-training distillation and find better configurations and interesting conclusions, such as larger student LLMs generally benefiting more from pre-training distillation, while a larger teacher LLM does not necessarily guarantee better results. We hope our exploration of the design space will inform future practices in pre-training distillation.', 'data_source': 'hf_paper', 'gattered_datetime': 'Mon, 21 Oct 2024 00:00:00 GMT', 'link': 'https://huggingface.co/papers/2410.16215', 'title': 'Pre-training Distillation for Large Language Models: A Design Space Exploration', 'uuid': '25f63475-6956-4f74-9d4a-5638caf5ba63'}, 'recommendation_log_uuid': '94d3cfe6-3a39-4d6d-90e8-4f83f642c38f'}, {'item': {'abstract': 'Formal proofs are challenging to write even for experienced experts. Recent progress in Neural Theorem Proving (NTP) shows promise in expediting this process. However, the formal corpora available on the Internet are limited compared to the general text, posing a significant data scarcity challenge for NTP. To address this issue, this work proposes Alchemy, a general framework for data synthesis that constructs formal theorems through symbolic mutation. Specifically, for each candidate theorem in Mathlib, we identify all invocable theorems that can be used to rewrite or apply to it. Subsequently, we mutate the candidate theorem by replacing the corresponding term in the statement with its equivalent form or antecedent. As a result, our method increases the number of theorems in Mathlib by an order of magnitude, from 110k to 6M. Furthermore, we perform continual pretraining and supervised finetuning on this augmented corpus for large language models. Experimental results demonstrate the effectiveness of our approach, achieving a 5% absolute performance improvement on Leandojo benchmark. Additionally, our synthetic data achieve a 2.5% absolute performance gain on the out-of-distribution miniF2F benchmark. To provide further insights, we conduct a comprehensive analysis of synthetic data composition and the training paradigm, offering valuable guidance for developing a strong theorem prover.', 'data_source': 'hf_paper', 'gattered_datetime': 'Mon, 21 Oct 2024 00:00:00 GMT', 'link': 'https://huggingface.co/papers/2410.15748', 'title': 'Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation', 'uuid': '6dd5f712-1eb0-412d-8986-5a15035aa66b'}, 'recommendation_log_uuid': '5abcd9f5-112c-4ca2-8d7f-5b76f7b88f02'}, {'item': {'abstract': 'With the advancements in open-source models, training (or finetuning) models on custom datasets has become a crucial part of developing solutions which are tailored to specific industrial or open-source applications. Yet, there is no single tool which simplifies the process of training across different types of modalities or tasks. We introduce AutoTrain (aka AutoTrain Advanced) -- an open-source, no code tool/library which can be used to train (or finetune) models for different kinds of tasks such as: large language model (LLM) finetuning, text classification/regression, token classification, sequence-to-sequence task, finetuning of sentence transformers, visual language model (VLM) finetuning, image classification/regression and even classification and regression tasks on tabular data. AutoTrain Advanced is an open-source library providing best practices for training models on custom datasets. The library is available at this https URL. AutoTrain can be used in fully local mode or on cloud machines and works with tens of thousands of models shared on Hugging Face Hub and their variations.', 'data_source': 'hf_paper', 'gattered_datetime': 'Mon, 21 Oct 2024 00:00:00 GMT', 'link': 'https://huggingface.co/papers/2410.15735', 'title': 'AutoTrain: No-code training for state-of-the-art models', 'uuid': '0a6d2f91-1525-44b6-91c7-5acbed770978'}, 'recommendation_log_uuid': '4790c2f2-d25b-4b6f-bf0f-8254a74dd144'}]
    items = []
    try:
        for entry in data:
            item_data = entry['item']
            item_data['recommendation_log_uuid'] = entry['recommendation_log_uuid']
            items.append(item_data)
        item = pd.DataFrame(items)
        #item = item.sort_values('title')
        item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    except:
        item = ''
    return item

def get_user_cliked(access_token):
    user = user_data(access_token)
    id = user['uuid'].iloc[0]
    headers = {
        "Authorization" : f'Bearer {access_token}'
    }
    response = requests.get(f"{ROOT}/api/user_history/{id}", headers=headers)
    try:
        data = response.json()
        item = pd.json_normalize(data['history'])
        #item = item.sort_values('item_title')
        item['clicked_time'] = item['clicked_time'].apply(format_date)
        item['item_date'] = item['item_date'].apply(format_date)
    except:
        item = ''
    return item

#獲取新聞
def item_data():
    response = requests.get(f'{ROOT}/api/item')
    items = response.json()
    items = pd.DataFrame(items['data'])
    item = items.sort_values('title')
    item['gattered_datetime'] = item['gattered_datetime'].apply(format_date)
    return item

def get_formatted_datetime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def click_data(access_token, link):
    time = get_formatted_datetime()
    recommend_item = get_recommend(access_token)
    unrecommend_item = get_unrecommend(access_token)
    try:
        item_content = recommend_item.loc[recommend_item['link'] == link]
    except:
        item_content = unrecommend_item.loc[unrecommend_item['link'] == link]
    item_id = item_content['uuid'].iloc[0]
    print(item_id)
    uuid = item_content['recommendation_log_uuid'].iloc[0]
    user = user_data(access_token)
    id = user['uuid'].iloc[0]
    data = {
        "user_id" : id,
        "item_id": item_id,
        "clicked_time": time
    }
    status = {
        "clicked": True
    }
    requests.put(f'{ROOT}:5000/api/recommend/{uuid}', json=status)
    requests.post(f'{ROOT}:5000/api/behavior', json = data)

