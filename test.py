import requests
import json
import hashlib
import hmac

# Define the API endpoint
url = "http://localhost:5000/api/user"  # Replace with the actual endpoint

# Data to be sent in the POST request
data = {
    "uuid": "32a575db-aac9-4e6a-bbd5-0ef29607a575",
    "account": "ruby543", 
    "password": "ruby", 
    "email": "ruby@example.com",
    "line_id": ""
}

# Convert the data to JSON
json_data = json.dumps(data)

# Prepare the secret key and signature
secret = '123'  # This should be the same secret used by the server
hash_object = hmac.new(secret.encode('utf-8'), msg=json_data.encode('utf-8'), digestmod=hashlib.sha256)
signature = "sha256=" + hash_object.hexdigest()

# Define headers, including the signature
headers = {
    "Content-Type": "application/json",
    "X-Fju-Signature-256": signature
}

# Make the POST request
response = requests.post(url, data=json_data, headers=headers)

# Print the response (status code and body)
print("Status Code:", response.status_code)
print("Response Body:", response.json())  # Assuming the response is JSON
