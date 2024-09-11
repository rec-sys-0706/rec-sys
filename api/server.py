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
API_KEY = "news"

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
   # elif request.method == 'DELETE':
    #    news_data.remove(news)
     #   return '', 204


@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users_data)


@app.route('/api/user-news-records', methods=['GET'])
def get_user_news_records():
    return jsonify(user_news_records)


if __name__ == '__main__':
    app.run(debug=True)
