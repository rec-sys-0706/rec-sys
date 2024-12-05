from datetime import datetime
from flask_caching import Cache
import base64
import logging
import uuid

# 初始化緩存（Redis）
cache = Cache(config={'CACHE_TYPE': 'RedisCache', 'CACHE_REDIS_URL': 'redis://localhost:6379/0'})

def dict_has_exact_keys(dictionary: dict, required_keys: list):
    dict_keys = set(dictionary.keys())
    req_keys = set(required_keys)
    logging.info(dict_keys)
    logging.info(req_keys)

    if dict_keys != req_keys:
        extra_keys = dict_keys - req_keys
        missing_keys = req_keys - dict_keys
        if extra_keys:
            logging.error(f"\033[33mExtra keys:\033[0m {extra_keys}")
        if missing_keys:
            logging.error(f"\033[31mMissing keys:\033[0m {missing_keys}")
        return False
    return True

def dict_has_keys():
    pass
# def validate_dict_keys(dictionary: dict, valid_keys: list):
#     """Check if the dictionary has no keys other than those in valid_keys"""
#     return all(key in valid_keys for key in dictionary.keys())

def validate_dict_keys(data: dict, headers: list):
    dict_keys = set(data.keys())
    valid_keys = set(headers)

    # 檢查請求中的字段是否都在模型字段之內
    if not dict_keys.issubset(valid_keys):
        extra_keys = dict_keys - valid_keys
        logging.error(f"Invalid keys in the request: {extra_keys}")
        return False
    return True

def format_date(date_obj):
    return datetime.strftime(date_obj, "%b %d, %Y")

def generate_random_scores(items: list[dict], users: list[dict]) -> list[dict]:
    recommendations = []
    
    for user in users:
        user_uuid = user['uuid']
        for item in items:
            item_uuid = item['uuid']
            # recommend_score = random.randint(0, 1)  
            recommendations.append({
                'uuid': str(uuid.uuid4()),
                'user_id': user_uuid,
                'item_id': item_uuid,
                'recommend_score': 0,
                # 'gattered_datetime': item['gattered_datetime']
            })
    
    return recommendations


# 通用緩存函數
def get_or_cache_item_image(item_uuid, base64_image):
    """
    檢查緩存中是否存在 item.image，如果不存在則進行緩存。
    :param item_uuid: Item 的唯一 UUID
    :param base64_image: 圖片的 Base64 編碼
    :return: 圖片的 Base64 編碼
    """
    cache_key = f"item_image:{item_uuid}"

    # 檢查是否存在緩存
    cached_image = cache.get(cache_key)
    if cached_image:
        return cached_image

    # 如果緩存不存在，添加到緩存
    cache.set(cache_key, base64_image, timeout=86400)  # 緩存 1 天
    return base64_image