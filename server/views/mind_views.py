import logging

from apiflask import APIBlueprint, abort
from flask import jsonify, request
from config import DB

from server.models.mind import Mind, MindSchema
from server.utils import validate_dict_keys



mind_blueprint = APIBlueprint('mind', __name__)

mind_schema = MindSchema()

headers = [column.name for column in Mind.__table__.columns]

@mind_blueprint.route('/', methods = ['POST'])
def create_mind_data():
        input = request.json
        if not validate_dict_keys(input, headers):
            abort(400)
        try:
            mind = Mind(**input)
            DB.session.add(mind)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 201
        except Exception as error:
            logging.error(f'[INSERT ERROR] - {error}')
            DB.session.rollback()
            abort(500) 

def split_list(data_list, chunk_size):
    """將清單分批處理"""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]

@mind_blueprint.route('check_mind_ids', methods=['POST'])
def check_mind_ids():
    data = request.json
    mind_ids = data.get('mind_ids', [])

    if not mind_ids:
        return jsonify({"msg": "No mind_ids provided"}), 400

    # 分批查詢，避免參數過多錯誤
    existing_mind_ids = set()
    chunk_size = 1000  # 每次查詢 1000 個參數
    for chunk in split_list(mind_ids, chunk_size):
        results = Mind.query.filter(Mind.mind_id.in_(chunk)).all()
        existing_mind_ids.update(result.mind_id for result in results)

    # 找出不在 mind table 的 mind_id
    missing_mind_ids = [mind_id for mind_id in mind_ids if mind_id not in existing_mind_ids]

    return jsonify({"missing_mind_ids": missing_mind_ids}), 200


@mind_blueprint.route('/get_item_uuid', methods=['GET'])
def get_item_uuid():
    mind_id = request.args.get('mind_id')

    # 檢查 mind_id 是否提供
    if not mind_id:
        return jsonify({"msg": "Missing mind_id parameter"}), 400

    # 查詢 item_uuid
    mind = Mind.query.filter_by(mind_id=mind_id).first()
    if mind:
        return jsonify({"item_uuid": str(mind.item_uuid)}), 200
    else:
        return jsonify({"msg": "Mind ID not found"}), 404