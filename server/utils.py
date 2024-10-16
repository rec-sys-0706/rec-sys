import logging
import hashlib
import hmac

def check_api_key(req) -> bool:
    secret = '123' # os.environ.get('SQL_SECRET')
    signature = req.headers.get('X-Fju-Signature-256')
    payload = req.data

    if signature is None:
        logging.error("No signature is provided.")
        return False

    hash_object = hmac.new(secret.encode('utf-8'), msg=payload, digestmod=hashlib.sha256)
    expected_signature = "sha256=" + hash_object.hexdigest()

    return hmac.compare_digest(expected_signature, signature)

def dict_has_exact_keys(dictionary: dict, required_keys: list):
    dict_keys = set(dictionary.keys())
    req_keys = set(required_keys)

    if dict_keys != req_keys:
        extra_keys = dict_keys - req_keys
        missing_keys = req_keys - dict_keys
        if extra_keys:
            logging.error(f"\033[33mExtra keys:\033[0m {extra_keys}")
        if missing_keys:
            logging.error(f"\033[31mMissing keys:\033[0m {missing_keys}")
        return False
    return True

def validate_dict_keys(dictionary: dict, valid_keys: list):
    """Check if the dictionary has no keys other than those in valid_keys"""
    return all(key in valid_keys for key in dictionary.keys())
