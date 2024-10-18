import logging
import functools
import uuid

from apiflask import APIBlueprint, Schema
from apiflask.fields import String
from flask import Blueprint, request, jsonify, abort
from flask_jwt_extended import create_access_token, jwt_required
from flask_sqlalchemy.model import Model
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from server.utils import dict_has_exact_keys, validate_dict_keys
from server.models.user import User
from config import DB

class HeaderSchema(Schema):
    signature = String(
        required=True,
        metatdata={'description': 'signature'}
    )

def generate_blueprint(model: Model,
                       schema: SQLAlchemyAutoSchema,
                       table_name: str,
                       input_schema) -> Blueprint:
    """Generate basic RESTful API endpoints by `table_name`"""
    blueprint = APIBlueprint(table_name, __name__)
    
    #從模型中提取數據表的所有列名
    headers = [column.name for column in model.__table__.columns]

    @blueprint.route('', methods=['GET'])
    @jwt_required()
    def read_data():
        rows = model.query.all()
        return jsonify({
            'data': [row.serialize() for row in rows]
        }), 200

    @blueprint.route('/<string:uuid>', methods=['GET'])
    @jwt_required()
    def read_data_by_uuid(uuid):
        row = model.query.get_or_404(uuid)
        return jsonify({
            'data': row.serialize()
        }), 200

    @blueprint.route('', methods=['POST'])
    @blueprint.input(input_schema, location='json')
    @jwt_required()
    def create_data():
        logging.info(f'[Received Data] {request.json}')
        data = schema.load(request.json, partial=True)
        data['uuid'] = str(uuid.uuid4())
        if not dict_has_exact_keys(data, headers):
            abort(400)
        try:
            item = model(**data)
            DB.session.add(item)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 201 # Created
        except Exception as error:
            logging.error(f'[INSERT ERROR] - {error}')
            DB.session.rollback()
            abort(500)

    @blueprint.route('/<string:uuid>', methods=['PUT'])
    @blueprint.input(input_schema(partial=True), location='json')
    @jwt_required()
    def update_data(uuid):
        item = model.query.get_or_404(uuid)
        data = schema.load(request.json, partial=True)
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
    @jwt_required()
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
