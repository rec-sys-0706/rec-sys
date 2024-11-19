import logging
import uuid

from apiflask import APIBlueprint
from sqlalchemy.exc import SQLAlchemyError
from flask import request, jsonify, abort
from server.utils import validate_dict_keys

from server.models.recommendation_log import Recommendationlog, RecommendationlogSchema
from config import DB

recommendation_bp = APIBlueprint('recommendation_log', __name__)

recommendation_schema = RecommendationlogSchema()

headers = [column.name for column in Recommendationlog.__table__.columns]

# @recommendation_bp.route('/model', methods = ['Post'])
# def create_recommend_items():
#         input = request.json
#         input['clicked'] = input.get('clicked', 0)
#         if not validate_dict_keys(input, headers):
#             abort(400)
#         try:
#             item = Recommendationlog(**input)
#             DB.session.add(item)
#             DB.session.commit()
#             return jsonify({'message': 'Success'}), 201
#         except Exception as error:
#             logging.error(f'[INSERT ERROR] - {error}')
#             DB.session.rollback()
#             abort(500) 

####once 1000
@recommendation_bp.route('/model', methods=['POST'])
def create_recommend_items():
    inputs = request.json  # 接收多筆資料，應該是列表格式
    # 驗證輸入格式是否為列表
    if not isinstance(inputs, list):
        return jsonify({'error': 'Input should be a list'}), 400

    # # 處理輸入資料
    # items_to_insert = []
    # for input in inputs:
    #     input['clicked'] = input.get('clicked', 0)
        
    #     # 驗證資料完整性
    #     if not validate_dict_keys(input, headers):
    #         return jsonify({'error': 'Invalid input data format'}), 400

    #     # 構建要插入的數據
    #     items_to_insert.append(input)

    # 分批次插入資料
    batch_size = 10000
    total_records = len(inputs)
    inserted_records = 0

    try:
        for i in range(0, total_records, batch_size):
            batch = inputs[i:i + batch_size]  # 每次取出 batch_size 筆資料

            # 檢查資料完整性並準備插入
            for record in batch:
                if not validate_dict_keys(record, headers):
                    return jsonify({'error': f'Invalid input data format in batch starting at index {i}'}), 400

            # 使用 bulk_insert_mappings 來批量插入資料
            DB.session.bulk_insert_mappings(Recommendationlog, batch)
            DB.session.commit()
            inserted_records += len(batch)

        return jsonify({'message': 'Success', 'inserted_records': inserted_records}), 201

    except SQLAlchemyError as error:
        logging.error(f'[INSERT ERROR] - {error}')
        DB.session.rollback()
        abort(500)

@recommendation_bp.route('/<uuid:recommend_uuid>', methods=['PUT'])
def update_recommendation_log(recommend_uuid):
    try:
        # get clikced_record
        input_data = request.get_json()
        # clicked = input_data.get('clicked', False)

        # search for recommendation_log_uuid
        recommendation_log = Recommendationlog.query.filter_by(uuid=recommend_uuid).first()

        if not recommendation_log:
            return jsonify({'message': 'Recommendation log not found'}), 404

        # update
        # recommendation_log.clicked = clicked
        DB.session.commit()

        return jsonify({'message': 'Recommendation log updated successfully'}), 200

    except Exception as error:

        logging.error(f'[UPDATE ERROR] - {error}')
        DB.session.rollback()
        return jsonify({'message': 'Failed to update recommendation log'}), 500
    



