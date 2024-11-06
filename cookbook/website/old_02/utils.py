import os
import hmac
import hashlib

# 已經被 flask_jwt_extended 取代
def get_signature(payload=''):
    # Get SQL_SECRET
    secret_key = os.environ.get('SQL_SECRET')
    # Compute the HMAC-SHA256 signature
    hash_object = hmac.new(secret_key.encode('utf-8'), msg = payload.encode('utf-8'), digestmod=hashlib.sha256)
    signature = "sha256=" + hash_object.hexdigest()
    return signature
# payload = '{"example": "data"}' # 如果是 GET 則不用payload
# Prepare the headers, including the x-hub-signature-256
