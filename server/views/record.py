import logging
from msilib import schema
import uuid
from apiflask import APIBlueprint, abort
from flask import request, jsonify
from flask_jwt_extended import jwt_required
from server.models.user import User
from server.models.behavior import Behavior
from server.models.item import Item
from config import DB
from server.utils import dict_has_exact_keys
from flask_sqlalchemy.model import Model

user_history_bp = APIBlueprint('user_hitory_bp', __name__)

@user_history_bp.route('/<string:user_uuid>', methods = ['GET'])
####測試的時候可以先把這個jwt註解掉
@jwt_required()
def read_browsing_history(user_uuid):
    history = Behavior.query.filter_by(user_id=user_uuid).order_by(Behavior.clicked_time.desc()).all()

    if not history:
        abort(404, description="No viewing history found for this user.")

    result = []
    for h in history:
        item = h.item
        history_data = {
            'item_id': h.item_id,
            'item_title': item.title,
            'item_date': item.gattered_datetime,
            'item_link': item.link,
            'clicked_time': h.clicked_time
        }
        result.append(history_data)

    return jsonify({'history': result}), 200

model: Model
headers = [column.name for column in model.__table__.columns]

@user_history_bp.route('/<string:user_uuid>', methods = ['POST'])
@jwt_required()
def create_behavior_db(user_uuid):
    logging.info(f'[Received Data] {request.json}')
    clicked_time = request.json['clicked_time']
    input = request.json
    del input['clicked_time']
    logging.info(input)
    data = schema.load(input, partial=True)
    data['clicked_time'] = clicked_time
    data['uuid'] = str(uuid.uuid4())
    if not dict_has_exact_keys(data, headers):
        logging.info(f'[Received Data] {data}')
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