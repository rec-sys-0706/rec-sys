import requests
from dotenv import load_dotenv

load_dotenv()

def send_broadcast_message(server_url, message_text):
    url = f"{server_url}/broadcast"
    data = {
        "message": message_text
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("廣播訊息已成功發送")
    except requests.exceptions.RequestException as e:
        print("發送廣播訊息時發生錯誤:", e)

if __name__ == "__main__":
    # 設定為正確的伺服器 URL，依需求使用 http 或 https
    server_url = "https://recsys.csie.fju.edu.tw"  # 遠端伺服器
    message_text = input("請輸入要廣播的訊息: ")
    send_broadcast_message(server_url, message_text)