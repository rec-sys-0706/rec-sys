import logging
import functools

from flask import Blueprint, request, jsonify, abort
from flask_sqlalchemy.model import Model

from server.utils import dict_has_exact_keys, validate_dict_keys, check_api_key
from config import DB


def authenticate(func):
    """Decorator for SQLError & api key checking"""
    @functools.wraps(func) # ? ChatGPT magic.
    def wrapper(*args, **kwargs):
        if not check_api_key(request):
            abort(403) # Permission denied.
        return func(*args, **kwargs)
    return wrapper




def generate_blueprint(model: Model, table_name: str) -> Blueprint:
    """Generate basic RESTful API endpoints by `table_name`"""
    blueprint = Blueprint(table_name, __name__)
    headers = [column.name for column in model.__table__.columns]

    @blueprint.route('', methods=['GET'])
    @authenticate
    def read_data():
        rows = model.query.all()
        return jsonify({
            'data': [row.serialize() for row in rows]
        }), 200

    @blueprint.route('/<string:uuid>', methods=['GET'])
    @authenticate
    def read_data_by_uuid(uuid):
        row = model.query.get_or_404(uuid)
        return jsonify({
            'data': row.serialize()
        }), 200

    @blueprint.route('', methods=['POST'])
    @authenticate
    def create_data():
        data = request.json
        if not dict_has_exact_keys(data, headers):
            abort(400)
        try:
            item = model(**data)
            DB.session.add(item)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 201 # Created
        except Exception as error:
            logging.error('INSERT ERROR')
            DB.session.rollback()
            abort(500)

    @blueprint.route('/<string:uuid>', methods=['PUT'])
    @authenticate
    def update_data(uuid):
        item = model.query.get_or_404(uuid)
        data = request.json
        if not validate_dict_keys(data, headers):
            abort(400)
        try:
            for key, value in data.items():
                setattr(item, key, value)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 204 # Modified
        except Exception as error:
            logging.error('UPDATE ERROR')
            DB.session.rollback()
            abort(500)

    @blueprint.route('/<string:uuid>', methods=['DELETE'])
    @authenticate
    def delete_data(uuid):
        item = model.query.get_or_404(uuid)
        try:
            DB.session.delete(item)
            DB.session.commit()
            return jsonify({'message': 'Deleted'}), 204
        except Exception as error:
            logging.error('UPDATE ERROR')
            DB.session.rollback()
            abort(500)

    return blueprint
