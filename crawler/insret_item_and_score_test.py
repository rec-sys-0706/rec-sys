import requests
import os
import pandas as pd
from training.recommend import recommend
import numpy as np

def post_news_and_score(item_data, users):
    
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
        
    recommendations = recommend(items, users)
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
                
    print("新聞更新完畢")
    
#post_news_and_score('cnn_news_output/cnn_news_2024-11-18_14-37-05.csv')
#post_news_and_score('cnn_news_output/cnn_news_2024-11-16_17-38-31.csv')
#post_news_and_score('cnn_news_output/cnn_news_2024-11-16_10-08-46.csv')
#post_news_and_score('cnn_news_output/cnn_news_2024-11-15_09-22-54.csv')
#post_news_and_score('cnn_news_output/cnn_news_2024-11-14_23-56-28.csv')
#post_news_and_score('cnn_news_output/cnn_news_2024-11-14_11-26-33.csv')