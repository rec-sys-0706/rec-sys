import requests
import os
import pandas as pd
from datetime import datetime


def get_users():    
    response = requests.get(f"{os.environ.get('ROOT')}:5000/api/user/test")
    data = response.json()
    
    users = []
    if 'general_users' in data:
        users.extend([{'uuid': user['uuid']} for user in data['general_users']])
    if 'u_users' in data:
        users.extend([{'uuid': user['uuid']} for user in data['u_users']])
    
    folder_path = 'user_data_folder'
    os.makedirs(folder_path, exist_ok=True)
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    user_data = {
        'uuid': [user['uuid'] for user in users]
    }  
    
    users_df = pd.DataFrame(user_data)
    users_df.to_csv(os.path.join(folder_path, f'user_data_{current_time}.csv'), index=False)
    
    return users