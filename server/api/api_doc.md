
## 概述
新聞和使用者相關的操作，允許你創建、讀取、更新和刪除（CRUD）新聞和使用者資料。

## 基本 URL
Authorization: Bearer <api_news>

## 方法清單

### 1. 獲取所有新聞
- **方法**: `GET`
- **路徑**: `/news`
- **說明**: 獲取所有新聞的資料。

#### 回應範例:
```json
[
    {
        "Id": 1,
        "Category": "Politics",
        "Subcategory": "Elections",
        "Title": "2024 Election Updates",
        "Abstract": "Latest updates on the 2024 elections.",
        "Content": "The 2024 elections are seeing a lot of activity...",
        "URL": "https://example.com/news/1",
        "PublicationDate": "2024-08-27T14:30:00Z"
    }
]
```
----------------------------------------------------------------------------

### 2. 創建一條新聞
方法: POST
路徑: /news
說明: 創建一條新的新聞記錄。
請求參數:
Category (string): 新聞類別。
Subcategory (string): 新聞子類別。
Title (string): 新聞標題。
Abstract (string): 新聞摘要。
Content (string): 新聞內容。
URL (string): 新聞的外部連結。
PublicationDate (string): 新聞發布的日期和時間（ISO格式）。
```json
請求範例:
{
    "Category": "Technology",
    "Subcategory": "AI",
    "Title": "New AI Developments",
    "Abstract": "Summary of new AI trends.",
    "Content": "Detailed content about AI.",
    "URL": "https://example.com/news/2",
    "PublicationDate": "2024-08-28T10:00:00Z"
}
```
回應範例:
```json
{
    "Id": 2,
    "Category": "Technology",
    "Subcategory": "AI",
    "Title": "New AI Developments",
    "Abstract": "Summary of new AI trends.",
    "Content": "Detailed content about AI.",
    "URL": "https://example.com/news/2",
    "PublicationDate": "2024-08-28T10:00:00Z"
}
```
----------------------------------------------------------------------------

3. 獲取特定ID的新聞
方法: GET
路徑: /news/<id>
說明: 根據ID獲取特定新聞的詳細資料。

回應範例:
```json
{
    "Id": 1,
    "Category": "Politics",
    "Subcategory": "Elections",
    "Title": "2024 Election Updates",
    "Abstract": "Latest updates on the 2024 elections.",
    "Content": "The 2024 elections are seeing a lot of activity...",
    "URL": "https://example.com/news/1",
    "PublicationDate": "2024-08-27T14:30:00Z"
}
```
--------------------------------------------------------------------------------------
4. 更新特定ID的新聞
方法: PUT
路徑: /news/<id>
說明: 更新特定ID的新聞資料。
請求範例:
```json
{
    "Title": "Updated AI Developments"
}
```
回應範例:
```json
{
    "Id": 2,
    "Category": "Technology",
    "Subcategory": "AI",
    "Title": "Updated AI Developments",
    "Abstract": "Summary of new AI trends.",
    "Content": "Detailed content about AI.",
    "URL": "https://example.com/news/2",
    "PublicationDate": "2024-08-28T10:00:00Z"
}
```
------------------------------------------------------------------------------------    
5. 刪除特定ID的新聞
方法: DELETE
路徑: /news/<id>
說明: 刪除特定ID的新聞記錄。
回應範例:
```json
{
    "message": "News record deleted successfully."
}
```
-----------------------------------------------------------------------------------
錯誤代碼
400 Bad Request: 表示請求資料有誤。
401 Unauthorized: 表示未提供有效的API密鑰。
404 Not Found: 表示請求的資源不存在。
500 Internal Server Error: 表示伺服器內部發生錯誤。
____________________________________________________________________________________


*****Flask API 端*****
```json
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# 假設的新聞數據
news_data = [
    {
        "Id": 1,
        "Category": "Politics",
        "Subcategory": "Elections",
        "Title": "2024 Election Updates",
        "Abstract": "Latest updates on the 2024 elections.",
        "Content": "The 2024 elections are seeing a lot of activity...",
        "URL": "https://example.com/news/1",
        "PublicationDate": "2024-08-27T14:30:00Z"
    },
]

# 假設的使用者數據
users_data = [
    {
        "UserID": 1,
        "UserName": "johndoe",
        "PasswordHash": "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3",
        "LineID": "JOHN123",
    },
]

# 假設的使用者新聞記錄數據
user_news_records = [
    {
        "UserID": 1,
        "NewsID": 1,
        "ViewedDate": "2024-08-27T15:00:00Z"
    },
]

# API Key 認證
API_KEY = "api_news"

def check_api_key(request):
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.split(" ")[1] == API_KEY:
        return True
    else:
        abort(401, description="Unauthorized")


@app.route('/api/news', methods=['GET', 'POST'])
def get_or_create_news():
    # 獲取新聞
    if request.method == 'GET':
        return jsonify(news_data)

    # 創建新新聞
    elif request.method == 'POST':
        new_news = request.json
        new_news['Id'] = len(news_data) + 1
        news_data.append(new_news)
        return jsonify(new_news), 201


@app.route('/api/news/<int:news_id>', methods=['PUT', 'DELETE'])
def update_or_delete_news(news_id):
    news = next((item for item in news_data if item['Id'] == news_id), None)

    if not news:
        abort(404, description="News not found")

    # 更新新聞
    if request.method == 'PUT':
        updated_data = request.json
        news.update(updated_data)
        return jsonify(news)

    # 刪除新聞
    elif request.method == 'DELETE':
        news_data.remove(news)
        return '', 204


@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users_data)


@app.route('/api/user-news-records', methods=['GET'])
def get_user_news_records():
    return jsonify(user_news_records)


if __name__ == '__main__':
    app.run(debug=True)
##http://127.0.0.1:5000/api/news 來獲取新聞數據。
##http://127.0.0.1:5000/api/users 來獲取使用者數據。
##http://127.0.0.1:5000/api/user-news-records 來獲取使用者新聞記錄
```
_________________________________________________________________

Python請求端:
```json
import requests

api_url = "http://127.0.0.1:5000/api/news"
headers = {
    "Authorization": "Bearer api_news"
}

# 獲取新聞列表
response = requests.get(api_url, headers=headers)
if response.status_code == 200:
    news_data = response.json()
    print("新聞列表:", news_data)
else:
    print(f"Error: {response.status_code}")

# 創建一條新新聞
new_news = {
    "Category": "Technology",
    "Subcategory": "AI",
    "Title": "New AI Developments",
    "Abstract": "Summary of new AI trends.",
    "Content": "Detailed content about AI.",
    "URL": "https://example.com/news/2",
    "PublicationDate": "2024-08-28T10:00:00Z"
}
response = requests.post(api_url, headers=headers, json=new_news)
if response.status_code == 201:
    created_news = response.json()
    print("創建的新新聞:", created_news)
else:
    print(f"Error: {response.status_code}")

# 更新一條新聞
update_news = {
    "Title": "Updated AI Developments"
}
response = requests.put(f"{api_url}/1", headers=headers, json=update_news)
if response.status_code == 200:
    updated_news = response.json()
    print("更新後的新聞:", updated_news)
else:
    print(f"Error: {response.status_code}")

# 刪除一條新聞
response = requests.delete(f"{api_url}/1", headers=headers)
if response.status_code == 204:
    print("刪除成功")
else:
    print(f"Error: {response.status_code}")
```