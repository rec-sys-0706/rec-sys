import requests
import os
import pandas as pd
from training.recommend import generate_random_scores
import numpy as np

def post_papers_and_score(item_data, users):
    
    df = pd.read_csv(item_data)
    df = df.drop(columns=['crawler_datetime', 'any_category'], errors='ignore')
    
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    
    api_item = f"{os.environ.get('ROOT')}:5000/api/item/crawler"
    
    items = []
    
    if api_item:  # 檢查環境變數是否存在
        
        for _, row in df.iterrows():
            json_data = row.to_dict()
        
            try:
                item_post = requests.post(api_item, json=json_data, timeout=10)
                
                if item_post.status_code == 201:
                    items.append(json_data)
                    print(f"API 發送成功: {item_post.text}")
                    
                if item_post.status_code != 201:
                    print(f"API 發送失敗: {item_post.text}")
                
            except requests.exceptions.RequestException as e:
                print(f"請求發生錯誤: {e}")    
    else:
        print("環境變數 ROOT 未設置")
        
    recommendations = generate_random_scores(items,users)
    api_recommendations = f"{os.environ.get('ROOT')}:5000/api/recommend/model"
    
    if api_recommendations:
        
        try:
            recommendations_post = requests.post(api_recommendations, json=recommendations, timeout=30) 
            if recommendations_post.status_code == 201:
                print(f"API 發送成功: {recommendations_post.text}")
                
            if recommendations_post.status_code != 201:
                print(f"API 發送失敗: {recommendations_post.text}")
                
        except requests.exceptions.RequestException as e:
                print(f"請求發生錯誤: {e}")
                
    else:
        print("環境變數 ROOT 未設置")
        
    print("論文更新完畢")
    
#post_papers_and_score('daily_papers_output/daily_papers_2024-11-08_01-26-26.csv')
#post_papers_and_score('daily_papers_output/daily_papers_2024-11-15_00-06-37.csv')
#post_papers_and_score('daily_papers_output/daily_papers_2024-11-18_14-36-27.csv')