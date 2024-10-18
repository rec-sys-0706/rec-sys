import logging
import hashlib
import hmac

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

def validate_dict_keys(dictionary: dict, valid_keys: list):
    """Check if the dictionary has no keys other than those in valid_keys"""
    return all(key in valid_keys for key in dictionary.keys())
