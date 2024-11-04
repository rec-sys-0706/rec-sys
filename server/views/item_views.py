from datetime import date, datetime
import logging

from apiflask import APIBlueprint
from flask import request, jsonify, abort
from flask_jwt_extended import jwt_required
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
    # today_items
    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    # today_items
    items = Item.query.filter(Item.gattered_datetime >= start_of_day, Item.gattered_datetime <= end_of_day).all()

    items_list = [{"id": item.uuid, 
                   "title": item.title, 
                   "date": item.gattered_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                   "link":item.data_source} for item in items]

    return jsonify(items_list), 200

