import requests
import os
import pandas as pd
from recommendation import generate_random_scores
from datetime import datetime
import numpy as np

def post_papers_and_score(item_data):
    response = requests.get(f"{os.environ.get('ROOT')}:5000/api/user/test")
    data = response.json()
    
    users = []
    if 'general_users' in data:
        users.extend([{'uuid': user['uuid']} for user in data['general_users']])
    if 'u_users' in data:
        users.extend([{'uuid': user['uuid']} for user in data['u_users']])
    
    folder_path = 'user_data_folder'
    os.makedirs(folder_path, exist_ok=True)
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    user_data = {
        'uuid': [user['uuid'] for user in users]
    }  
    
    users_df = pd.DataFrame(user_data)
    users_df.to_csv(os.path.join(folder_path, f'user_data_{current_time}.csv'), index=False)
    
    df = pd.read_csv(item_data)
    df = df.drop(columns=['crawler_datetime', 'any_category'], errors='ignore')
    
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    
    api_item = f"{os.environ.get('ROOT')}:5000/api/item/crawler"
    
    if api_item:  # 檢查環境變數是否存在
        
        for _, row in df.iterrows():
            json_data = row.to_dict()
            items = [json_data]
            try:
                item_post = requests.post(api_item, json=json_data, timeout=10)
                
                if item_post.status_code == 201:
                    recommendations = generate_random_scores(items,users)
                    api_recommendations = f"{os.environ.get('ROOT')}:5000/api/recommend/model"
                    recommendations_post = requests.post(api_recommendations, json=recommendations, timeout=30) 
                    if recommendations_post.status_code == 201:
                        print(f"API 發送成功: {recommendations_post.text}")
                
                if item_post.status_code != 201:
                    print(f"API 發送失敗: {item_post.text}")
                
            except requests.exceptions.RequestException as e:
                print(f"請求發生錯誤: {e}")
    else:
        print("環境變數 ROOT 未設置")
        
    print("論文更新完畢")
    
post_papers_and_score('daily_papers_output/daily_papers_2024-11-16_10-08-18.csv')