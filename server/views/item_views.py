from datetime import date, datetime
import logging

from apiflask import APIBlueprint
from datetime import timedelta
from flask import request, jsonify, abort
from flask_jwt_extended import jwt_required
from sqlalchemy import desc
from server.utils import validate_dict_keys

from server.models.item import Item, ItemSchema
from config import DB

item_blueprint = APIBlueprint('item', __name__)

item_schema = ItemSchema()

headers = [column.name for column in Item.__table__.columns]

### get_all_data
@item_blueprint.route('', methods=['GET'])
def read_items():
    rows = Item.query.all()
    return jsonify({
        'data': [row.serialize() for row in rows]
    }), 200

### get_certain_data
@item_blueprint.route('/<string:uuid>', methods=['GET'])
@jwt_required()
def read_item_by_uuid(uuid):
    logging.info(uuid)
    row = Item.query.get_or_404(uuid)
    return jsonify({
        'data': row.serialize()
    }), 200

### crawler
@item_blueprint.route('/crawler', methods = ['POST'])
def create_crawler_data():
        input = request.json
        if not validate_dict_keys(input, headers):
            abort(400)
        try:
            item = Item(**input)
            DB.session.add(item)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 201
        except Exception as error:
            logging.error(f'[INSERT ERROR] - {error}')
            DB.session.rollback()
            abort(500) 

### get today_items
@item_blueprint.route('/today', methods=['GET'])
def get_today_items():
    # 獲取 data_source 查詢參數
    data_source = request.args.get('data_source', None)

    # 定義要篩選的來源列表
    sources = []
    if data_source == 'news':
        sources = ['mit_news', 'cnn_news', 'bbc_news']
    elif data_source == 'papers':
        sources = ['hf_paper']
    elif data_source:
        sources = [data_source]

    # 查找符合條件的最近日期
    latest_item_query = Item.query
    if sources:
        latest_item_query = latest_item_query.filter(Item.data_source.in_(sources))

    latest_item = latest_item_query.order_by(desc(Item.gattered_datetime)).first()

    # 如果沒有找到符合條件的項目，返回空列表
    if not latest_item:
        return jsonify([]), 200

    # 計算從最近日期往回數七天的時間範圍
    end_date = latest_item.gattered_datetime
    start_date = end_date - timedelta(days=7)

    # 查詢符合時間範圍和 data_source 的項目
    items_query = Item.query.filter(
        Item.gattered_datetime >= start_date,
        Item.gattered_datetime <= end_date
    )

    if sources:
        items_query = items_query.filter(Item.data_source.in_(sources))

    items = items_query.order_by(desc(Item.gattered_datetime)).limit(20).all()

    # 構建結果列表
    items_list = [
        {
            "title": item.title,
            "abstract": item.abstract,
            "gattered_datetime": item.gattered_datetime,
            "link": item.link,
            "data_source": item.data_source
        }
        for item in items
    ]

    return jsonify(items_list), 200

