from datetime import date, datetime, time
import logging

from apiflask import APIBlueprint
from datetime import timedelta
from flask import request, jsonify, abort
from flask_jwt_extended import jwt_required
from sqlalchemy import desc
from server.utils import format_date, validate_dict_keys

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

    # # 計算本週的開始和結束時間
    # today = datetime.today() + timedelta(days=1)
    # start_of_week = today - timedelta(days=8)  # 上週的時間

    # 獲取參數中的日期（格式：YYYY-MM-DD）
    date_str = request.args.get('date', None)
    if not date_str:
        return jsonify({'error': 'Date parameter is required in format YYYY-MM-DD'}), 400

    try:
        # 將字符串轉換為日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # 定義目標日期的開始和結束時間
    start_of_day = datetime.combine(target_date, time.min)  # 2024-11-24 00:00:00
    end_of_day = datetime.combine(target_date, time.max)  # 2024-11-24 23:59:59

    # 查詢符合時間範圍和 data_source 的項目
    items_query = Item.query.filter(
        Item.gattered_datetime >= start_of_day,
        Item.gattered_datetime <= end_of_day  
    )

    if sources:
        items_query = items_query.filter(Item.data_source.in_(sources))

    # # 按時間排序並限制為前 10 筆
    # items = items_query.order_by(desc(Item.gattered_datetime)).all()

    # 檢查結果是否為空
    if not items_query:
        return 404

    # 構建結果列表
    items_list = [
        {
            "title": item.title,
            "category": item.category,
            "abstract": item.abstract,
            "gattered_datetime": format_date(item.gattered_datetime),
            "link": item.link,
            "data_source": item.data_source,
            "image": item.image
        }
        for item in items_query
    ]

    return jsonify(items_list), 200

