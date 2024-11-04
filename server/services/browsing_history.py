import logging
from datetime import datetime, timedelta
from apiflask import APIBlueprint, abort
from flask import jsonify, request
from flask_jwt_extended import jwt_required

from config import DB
from server.models.behavior import Behavior
from server.models.recommendation_log import Recommendationlog
from server.models.item import Item

user_history_bp = APIBlueprint('user_history_bp', __name__)


###瀏覽紀錄API
@user_history_bp.route('/<string:user_uuid>', methods=['GET'])
@jwt_required()
def read_browsing_history(user_uuid):
    # searching_browsing_history: descending order
    history = Behavior.query.filter_by(user_id=user_uuid).order_by(Behavior.clicked_time.desc()).all()

    if not history:
        abort(404, description="No viewing history found for this user.")

    result = []
    for h in history:
        item = h.item  # through behavior model to get item
        history_data = {
            'item_id': h.item_id,
            'item_title': item.title,
            'item_date': item.gattered_datetime,
            'item_link': item.link,
            'clicked_time': h.clicked_time
        }
        result.append(history_data)

    return jsonify({'history': result}), 200


@user_history_bp.route('/recommend/<uuid:user_uuid>', methods = ['GET'])
def get_recommend_items(user_uuid):
    try:

        data_source = request.args.get('data_source', None)

        recommendation_logs_query = Recommendationlog.query.filter_by(
            user_id=user_uuid,
            recommend_score=True
        )

        recommendation_logs = recommendation_logs_query.all()

        if not recommendation_logs:
            return jsonify({'message': 'No recommendations found'}), 404
        
        # 根據 recommendation_log 中的 item_id 查詢對應的 item 資料，並返回 recommendation_log 的 uuid
        result = []
        for log in recommendation_logs:
            item_query = Item.query.filter_by(uuid=log.item_id)
            if data_source:
                item_query = item_query.filter_by(data_source=data_source)

            item = item_query.order_by(Item.gattered_datetime.desc()).limit(10).all()
            if item:
                result.extend({
                    'recommendation_log_uuid': log.uuid,
                    'item': i.serialize()
                } for i in item)


        return jsonify(result[:10]), 200

    except Exception as error:
        logging.error(f'[GET ERROR] - {error}')
        abort(500)


@user_history_bp.route('/unrecommend/<uuid:user_uuid>', methods = ['GET'])
def get_unrecommend_items(user_uuid):
    try:
        data_source = request.args.get('data_source', None)

        recommendation_logs_query = Recommendationlog.query.filter_by(
            user_id = user_uuid,
            recommend_score = False
        )

        recommendation_logs = recommendation_logs_query.all()

        if not recommendation_logs:
            return jsonify({'message': 'No recommendations found'}), 404
        
        # 根據 recommendation_log 中的 item_id 查詢對應的 item 資料，並返回 recommendation_log 的 uuid
        result = []
        for log in recommendation_logs:
            item_query = Item.query.filter_by(uuid=log.item_id)
            if data_source:
                item_query = item_query.filter_by(data_source=data_source)

            item = item_query.order_by(Item.gattered_datetime.desc()).limit(10).all()
            if item:
                result.extend({
                    'recommendation_log_uuid': log.uuid,
                    'item': i.serialize()
                } for i in item)

        return jsonify(result[:10]), 200

    except Exception as error:
        logging.error(f'[GET ERROR] - {error}')
        abort(500)


####當週推薦的新聞
@user_history_bp.route('/recommend_week/<uuid:user_uuid>', methods=['GET'])
def get_week_recommendations(user_uuid):
    try:
        # this week
        today = datetime.today() + timedelta(days=1)
        start_of_week = today - timedelta(days=7)

        recommendation_logs = DB.session.query(Recommendationlog, Item)\
            .join(Item, Recommendationlog.item_id == Item.uuid)\
            .filter(
                Recommendationlog.user_id == user_uuid,
                Recommendationlog.recommend_score == True,
                Item.gattered_datetime >= start_of_week,
                Item.gattered_datetime <= today
            ).order_by(Item.gattered_datetime).limit(10).all()

        if not recommendation_logs:
            return jsonify({'message': 'No recommendations found for this week'}), 404
        
        result = [{
            'recommendation_log_uuid': log.uuid,
            'item': item.serialize()
        } for log, item in recommendation_logs]
        
        return jsonify(result), 200

    except Exception as error:
        logging.error(f'[GET ERROR] - {error}')
        abort(500)


####當週不推的新聞
@user_history_bp.route('/unrecommend_week/<uuid:user_uuid>', methods=['GET'])
def get_week_unrecommendations(user_uuid):
    try:
        # this week
        today = datetime.today() + timedelta(days=1)
        start_of_week = today - timedelta(days=7)

        recommendation_logs = DB.session.query(Recommendationlog, Item)\
            .join(Item, Recommendationlog.item_id == Item.uuid)\
            .filter(
                Recommendationlog.user_id == user_uuid,
                Recommendationlog.recommend_score == False,
                Item.gattered_datetime >= start_of_week,
                Item.gattered_datetime <= today
            ).order_by(Item.gattered_datetime).limit(10).all()

        if not recommendation_logs:
            return jsonify({'message': 'No recommendations found for this week'}), 404
        
        result = [{
            'recommendation_log_uuid': log.uuid,
            'item': item.serialize()
        } for log, item in recommendation_logs]
        
        return jsonify(result), 200
    
    except Exception as error:
        logging.error(f'[GET ERROR] - {error}')
        abort(500)

