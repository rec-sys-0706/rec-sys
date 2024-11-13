import requests
import os

# clicked_record = {
#     "user_id": "AFE8D734-A76E-4521-A2EE-B5234E79A480",
#     "item_id": "FB927645-2313-4397-B711-0078574D5888",
#     "clicked_time": "2024-10-20 10:30:00"
# }

# id = "AFE8D734-A76E-4521-A2EE-B5234E79A480"

# clicked_record = requests.post(f"{os.environ.get('ROOT')}:5000/api/behavior", json=clicked_record)

# print(clicked_record.content)

# history_list = requests.get(f"{os.environ.get('ROOT')}:5000/api/user_history/DDD82E79-0D8A-413D-9BDD-BCBB5910D28B")
# print(history_list.content)

# clicked_record = {
#     "user_id": "CEF6C498-2389-4C5D-AD22-57ED7131FD2E",
#     "item_id": "0FA5B2CC-5E22-42EE-8AEA-032EBAA54832",
#     "clicked_time": "2024-11-08 01:16:00"
# }
# clicked_record = requests.post(f"{os.environ.get('ROOT')}/api/behavior", json=clicked_record)
# print(clicked_record.content)

# data_source = "news"
# user_id = "D8391D38-F194-4701-B8EE-AC22F6653CE9"
# r_data = requests.get(f"{os.environ.get('ROOT')}/api/user_history/unrecommend/{user_id}?data_source={data_source}")
# r_data = r_data.json()
# for i in r_data:
#     print(i['item']['gattered_datetime'], i['item']['data_source'])

# fake_data = {
#     "item_uuid": "96E77024-2F27-4F8B-9A4E-0001640E79DC",
#     "mind_id": "N55528"
# }

# r_data = requests.post(f"{os.environ.get('ROOT')}/api/mind", json=fake_data)
# print(r_data.content)

News_id = 'N49435'
item_uuid = requests.get(f"{os.environ.get('ROOT')}/api/mind/get_item_uuid?mind_id={News_id}")
print(item_uuid.json()['item_uuid'])

User_accout = 'alice123'
user_uuid = requests.get(f"{os.environ.get('ROOT')}/api/user/get_uuid_by_account?account={User_accout}")
print(user_uuid.json()['user_uuid'])




