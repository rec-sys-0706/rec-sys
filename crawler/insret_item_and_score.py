import requests
import os
import pandas as pd
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
    
    if api_item:  # 檢查環境變數是否存在
        
        for _, row in df.iterrows():
            json_data = row.to_dict()
            items = [json_data]
            try:
                item_post = requests.post(api_item, json=json_data, timeout=10)
                
                if item_post.status_code == 201:
                    recommendations = recommend(items, users)
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
        
    print("新聞更新完畢")