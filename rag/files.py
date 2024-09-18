import io
import json

# 你的字典列表或數據對象
data_list = [
    {'key1': 'value1', 'key2': 'value2'},
    {'key1': 'value3', 'key2': 'value4'},
    # 根據需要添加更多字典
]

# 將字典列表轉換為 JSONL 字串
jsonl_data = '\n'.join([json.dumps(record) for record in data_list]).encode('utf-8')

# 將 JSONL 字串轉換為二進制類文件對象
file_like_object = io.BytesIO(jsonl_data)

# 上傳類文件對象並設置所需的檔名和用途
batch_input_file = client.files.create(
    file=('desired_filename.jsonl', file_like_object, 'application/json'),
    purpose='batch'
)
