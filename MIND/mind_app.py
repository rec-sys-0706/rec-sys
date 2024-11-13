import pandas as pd
import uuid
import requests
import os
import time

columns = ["News ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title Entities", "Abstract Entities"]

# # 讀取 news.tsv 檔案
# news_df = pd.read_csv("MIND/validation/news.tsv", sep='\t', names=columns)
# # print("資料筆數：", len(news_df))
# print(len(news_df))
# print(len(news_df['News ID'].unique()))

# # 設定收集時間
# gattered_datetime = "2023-01-01"  # API 接收的日期格式，使用 ISO 格式

# import pandas as pd

# 讀取 training 和 validation 的 news.tsv
training_news_df = pd.read_csv("MIND/training/news.tsv", sep='\t', names=columns)
validation_news_df = pd.read_csv("MIND/validation/news.tsv", sep='\t', names=columns)

# 合併兩個 DataFrame
combined_news_df = pd.concat([training_news_df, validation_news_df], ignore_index=True)

# 查看合併後的資料
print("合併後的資料筆數：", len(combined_news_df))
print(len(combined_news_df['News ID'].unique()))
# print(combined_news_df.head())


# # 遍歷每筆資料
# for index, row in news_df.iterrows():
#     # 在本地生成 UUID
#     item_uuid = str(uuid.uuid4())  # 確保 UUID 是字串格式

#     # 處理缺失值，如果 Abstract 為 NaN，則設為空字串
#     title = row['Title'] if pd.notna(row['Title']) else ""
#     abstract = row['Abstract'] if pd.notna(row['Abstract']) else ""

#     # Item 資料
#     item_data = {
#         "uuid": item_uuid,
#         "title": title,
#         "category": row['Category'],
#         "abstract": abstract,
#         "link": None,  # 如果你的 API 接收這個欄位
#         "data_source": "mind_small",
#         "gattered_datetime": gattered_datetime
#     }

#     # 發送 POST 請求到 items API
#     item_response = requests.post(f"{os.environ.get('ROOT')}/api/item/crawler", json=item_data)
#     if item_response.status_code == 201:
#         print(f"成功存入 item：{item_uuid}")

#         # Mind 資料
#         mind_data = {
#             "item_uuid": item_uuid,
#             "mind_id": row['News ID']
#         }

#         # 只有當 item 存入成功時才存入 mind
#         mind_response = requests.post(f"{os.environ.get('ROOT')}/api/mind", json=mind_data)
#         if mind_response.status_code == 201:
#             print(f"成功存入 mind：{item_uuid}")
#         else:
#             print(f"存入 mind 失敗：{item_uuid}, 錯誤：{mind_response.text}")
#     else:
#         print(f"存入 item 失敗：{item_uuid}, 錯誤：{item_response.text}")

