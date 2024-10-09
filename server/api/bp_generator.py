import logging
import functools

from flask import Blueprint, request, jsonify, abort
import pandas as pd

from server.api.sql_query import SQLError, select_all, select_by, insert, update, delete, get_headers
from server.utils import dict_has_exact_keys, validate_dict_keys, check_api_key



def catch_sql_error(func):
    """Decorator for SQLError"""
    @functools.wraps(func) # ? ChatGPT magic.
    def wrapper(*args, **kwargs):
        if not check_api_key(request):
            abort(403) # Permission denied.
        try:
            return func(*args, **kwargs)
        except Exception as error:
            logging.error(error)
            abort(500)
    return wrapper


def generate_blueprint(table_name):
    """Generate basic endpoints by `table_name`"""
    blueprint = Blueprint(table_name, __name__)
    headers = get_headers(table_name)


    @blueprint.route('', methods=['GET'])
    @catch_sql_error
    def read_data():
        rows = select_all(table_name)
        df = pd.DataFrame.from_records(rows, columns=headers)
        df = df.set_index('uuid')
        return jsonify({
            'data': df.to_dict(orient='index')
        }), 200

    @blueprint.route('/<string:uuid>', methods=['GET'])
    @catch_sql_error
    def read_data_by_uuid(uuid):
        row = select_by(table_name, ['uuid'], [uuid])
        if len(row):
            return jsonify({
                'data': dict(zip(headers, row[0]))
            }), 200
        else:
            return 'No this data', 404

    @blueprint.route('', methods=['POST'])
    @catch_sql_error
    def create_data():
        data = request.json
        if not dict_has_exact_keys(data, headers):
            abort(400)
        insert(table_name, data)
        return "success", 201 # Created.

    @blueprint.route('/<string:uuid>', methods=['PUT'])
    @catch_sql_error
    def update_data(uuid):
        data = request.json
        if not validate_dict_keys(data, headers):
            abort(400)
        update(table_name, 'uuid', uuid, data)
        return "success", 204


    @blueprint.route('/<string:uuid>', methods=['DELETE'])
    @catch_sql_error
    def delete_data(uuid):
        delete(table_name, ['uuid'], [uuid])
        return "success", 204

    return blueprint
