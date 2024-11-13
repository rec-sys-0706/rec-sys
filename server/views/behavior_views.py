import logging
import uuid

from apiflask import APIBlueprint
from flask import request, jsonify, abort
from server.utils import validate_dict_keys

from server.models.behavior import Behavior, BehaviorSchema
from config import DB

behavior_blueprint = APIBlueprint('behavior', __name__)

behavior_schema = BehaviorSchema()

headers = [column.name for column in Behavior.__table__.columns]

####點擊紀錄
@behavior_blueprint.route('', methods=['POST'])
# @jwt_required()
def create_behavior_db():
    logging.info(f'[Received Data] {request.json}')
    input_data = request.json

    # 查詢是否已存在相同的點擊紀錄
    existing_behavior = DB.session.query(Behavior).filter(
        Behavior.user_id == input_data['user_id'],
        Behavior.item_id == input_data['item_id']
    ).first()

    # 如果找到已有的紀錄，先刪除舊的紀錄
    if existing_behavior:
        try:
            DB.session.delete(existing_behavior)
            DB.session.commit()
        except Exception as error:
            logging.error(f'[DELETE ERROR] - {error}')
            DB.session.rollback()
            abort(500)

    # 準備插入新的點擊記錄
    input_data['uuid'] = str(uuid.uuid4())
    
    if not validate_dict_keys(input_data, headers):
        logging.info(f'[Received Data] {input_data}')
        abort(400)

    try:
        # 插入新的點擊記錄
        item = Behavior(**input_data)
        DB.session.add(item)
        DB.session.commit()
        return jsonify({'message': 'Success'}), 201  # Created
    except Exception as error:
        logging.error(f'[INSERT ERROR] - {error}')
        DB.session.rollback()
        abort(500)
    

