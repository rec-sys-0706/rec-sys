import requests
import os
import pandas as pd
from recommendation import generate_random_scores

def post_news_and_score(item_data, users):
    
    df = pd.read_csv(item_data)
    df = df.drop(columns=['crawler_datetime', 'any_category'], errors='ignore')
    
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
        
    recommendations = generate_random_scores(items, users)
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
    
#post_news_and_score('cnn_news_output/cnn_news_2024-11-14_11-26-33.csv')