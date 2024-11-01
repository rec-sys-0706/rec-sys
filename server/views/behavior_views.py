import logging
import uuid

from apiflask import APIBlueprint
from flask import request, jsonify, abort
from server.utils import validate_dict_keys

from server.models.behavior import Behavior, BehaviorSchema
from config import DB
from datetime import datetime
from sqlalchemy import func, Date

behavior_blueprint = APIBlueprint('behavior', __name__)

behavior_schema = BehaviorSchema()

headers = [column.name for column in Behavior.__table__.columns]

####點擊紀錄
@behavior_blueprint.route('', methods=['POST'])
# @jwt_required()
def create_behavior_db():
    logging.info(f'[Received Data] {request.json}')
    input_data = request.json
    clicked_time_dt = datetime.strptime(input_data['clicked_time'], '%Y-%m-%d %H:%M:%S')

    # 查詢是否已有當天的點擊紀錄
    # 有紀錄是true, 沒有則是None
    existing_behavior = DB.session.query(Behavior).filter(
        Behavior.user_id == input_data['user_id'],
        Behavior.item_id == input_data['item_id'],
        func.cast(Behavior.clicked_time, Date) == func.cast(input_data['clicked_time'], Date)  # 比較同一天
    ).first()

    # 如果沒有當天的點擊紀錄，才插入新的點擊記錄
    if not existing_behavior:
        input_data['uuid'] = str(uuid.uuid4())
        
        if not validate_dict_keys(input_data, headers):
            logging.info(f'[Received Data] {input_data}')
            abort(400)

        try:
            item = Behavior(**input_data)
            DB.session.add(item)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 201  # Created
        except Exception as error:
            logging.error(f'[INSERT ERROR] - {error}')
            DB.session.rollback()
            abort(500)
    else:
        # 如果有當天的點擊紀錄，返回已存在的訊息
        return jsonify({'message': 'Click already recorded for today'}), 200
    

