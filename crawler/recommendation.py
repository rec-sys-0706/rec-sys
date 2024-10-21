import random
import uuid
from datetime import datetime

def generate_random_scores(items: list[dict], users: list[dict]) -> list[dict]:
    recommendations = []
    
    for user in users:
        user_uuid = user['uuid']
        for item in items:
            item_uuid = item['uuid']
            recommend_score = random.randint(0, 1)  
            recommendations.append({
                'uuid': str(uuid.uuid4()),
                'user_uuid': user_uuid,
                'item_uuid': item_uuid,
                'recommend_score': recommend_score,
                'recommend_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return recommendations