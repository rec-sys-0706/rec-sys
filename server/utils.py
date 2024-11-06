import logging
import uuid

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
                'gattered_datetime': item['gattered_datetime']
            })
    
    return recommendations
