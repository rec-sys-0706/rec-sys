import random
import uuid

def generate_random_scores(items: list[dict], users: list[dict]) -> list[dict]:
    recommendations = []
    
    for user in users:
        user_uuid = user['uuid']
        for item in items:
            item_uuid = item['uuid']
            item_gattered_datetime = item['gattered_datetime']
            recommend_score = random.randint(0, 1)  
            recommendations.append({
                'uuid': str(uuid.uuid4()),
                'user_id': user_uuid,
                'item_id': item_uuid,
                'recommend_score': recommend_score,
                'gattered_datetime': item_gattered_datetime,
                'clicked': 0
            })
    
    return recommendations


