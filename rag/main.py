# from agent import batch_classify
# import logging
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
#     batch_classify('./rag/data/papers/combined.csv', './rag/batchinput.jsonl')


from typing import cast
esponse_format = cast(type, response_format)






# import logging
# from agent import ClassifierAgent
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
# agent = ClassifierAgent('rag/data/papers/combined.csv', 'rag/data/papers/result.csv')
# agent.batchify()
# # import io
# # import json
# from openai import OpenAI
# client = OpenAI()
# # # Your dictionary data
# # data_list = [
# #     {'key1': 'value1', 'key2': 'value2'},
# #     {'key1': 'value3', 'key2': 'value4'},
# #     # Add more dictionaries as needed
# # ]
# # # Convert dictionary to a JSON string
# # jsonl_data = '\n'.join([json.dumps(record) for record in data_list]).encode('utf-8')

# # # Convert JSON string to a file-like object
# # file_like_object = io.BytesIO(jsonl_data)

# # batch_input_file = client.files.create(
# #     file=('desired_filename.jsonl', file_like_object, 'application/json'),
# #     purpose='batch'
# # )

# batch_input_file_id = 'file-wZ7omSj37o0Z2QltLoyZsNwC'

# client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#       "description": "nightly eval job"
#     }
# )
