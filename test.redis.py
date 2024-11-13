import redis

try:
    # 連接到 Redis 服務器
    client = redis.Redis(host='localhost', port=6379)

    # 嘗試設置和獲取鍵值，檢查 Redis 是否正常工作
    client.set("test_key", "Hello Redis!")
    value = client.get("test_key")

    if value:
        print("Redis is running. Test value:", value.decode('utf-8'))
    else:
        print("Unable to retrieve test value from Redis.")

except redis.ConnectionError:
    print("Could not connect to Redis. Please ensure the server is running.")
