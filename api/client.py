import requests

# API URL和頭信息（包含API Key）
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
