import logging
from datetime import datetime, time, timedelta
from apiflask import APIBlueprint, abort
from flask import current_app, jsonify, request
from flask_jwt_extended import jwt_required

from config import DB
from server.models.behavior import Behavior
from server.models.recommendation_log import Recommendationlog
from server.models.item import Item
from server.utils import format_date

user_history_bp = APIBlueprint('user_history_bp', __name__)


###瀏覽紀錄API
@user_history_bp.route('/<string:user_uuid>', methods=['GET'])
@jwt_required()
def read_browsing_history(user_uuid):

    # 獲取 data_source 查詢參數
    data_source = request.args.get('data_source', None)
    # 初始查詢條件：過濾出 user 的瀏覽紀錄，按時間排序
    history_query = Behavior.query\
        .filter_by(user_id=user_uuid)\
        .join(Item, Behavior.item_id == Item.uuid)\
        .order_by(
            Behavior.clicked_time.desc(),  # 首先按 clicked_time 降序排列
            Item.category.asc()           # 然後按 category 升序排列
        )

    # 根據 data_source 過濾紀錄
    if data_source == 'news':
        # 當 data_source 為 'news' 時，包含多個來源
        history_query = history_query.filter(Item.data_source.in_(['mit_news', 'cnn_news', 'bbc_news', 'mind_small']))
    elif data_source == 'papers':
        # 當 data_source 為 'papers' 時，只篩選 'hf_paper'
        history_query = history_query.filter(Item.data_source == 'hf_paper')

    # 查詢並檢查結果
    history = history_query.all()

    if not history:
        abort(404, message="No viewing history found for this user.")

    result = []
    for h in history:
        item = h.item  # through behavior model to get item
        history_data = {
            'title': item.title,
            'category': item.category,
            'abstract': item.abstract,
            'data_source': item.data_source,
            'gattered_datetime': format_date(item.gattered_datetime),
            'link': item.link,
            'clicked_time': format_date(h.clicked_time),
        }
        result.append(history_data)

    return jsonify({'history': result}), 200


@user_history_bp.route('/recommend/<uuid:user_uuid>', methods = ['GET'])
def get_recommend_items(user_uuid):
    try:
        # # this week
        # today = datetime.today() + timedelta(days=1)
        # # today = datetime(2024, 10, 22) + timedelta(days=1)
        # start_of_week = today - timedelta(days=8)

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

        # 獲取參數中的 data_source
        data_source = request.args.get('data_source', None)

        is_recommend = request.args.get('is_recommend', None)

        # 檢查 is_recommend 是否為有效值
        if is_recommend not in ['true', 'false']:
            return jsonify({'error': 'is_recommend parameter must be "true" or "false"'}), 400

        is_recommend = is_recommend.lower() == 'true'
        
        # 查詢條件：user_id, recommend_score，並預過濾 data_source
        recommendation_logs_query = DB.session.query(Recommendationlog, Item)\
            .join(Item, Recommendationlog.item_id == Item.uuid)\
            .filter(
                Recommendationlog.user_id == user_uuid,
                Recommendationlog.is_recommend == is_recommend,
                Item.gattered_datetime >= start_of_day,
                Item.gattered_datetime <= end_of_day,
            )
        
        # 判斷 data_source 的值
        if data_source == 'news':
            # 如果 data_source 是 'news'，篩選多個來源
            recommendation_logs_query = recommendation_logs_query.filter(
                Item.data_source.in_(['mit_news', 'cnn_news', 'bbc_news'])
            )
        elif data_source == 'papers':
            # 如果 data_source 是 'papers'，篩選 hf_paper
            recommendation_logs_query = recommendation_logs_query.filter(Item.data_source == 'hf_paper')
        
        # 排序
        if is_recommend:
            recommendation_logs = recommendation_logs_query\
                .order_by(
                    # Item.gattered_datetime.desc(),
                    Recommendationlog.recommend_score.desc()
                )\
                .all()
        else:
            recommendation_logs = recommendation_logs_query\
                .order_by(
                    # Item.gattered_datetime.desc(),
                    Item.title.asc()
                )\
                .all()


        # 檢查結果是否為空
        if not recommendation_logs:
            return jsonify({'message': 'No recommendations found for the specified source'}), 404

        # 構建結果列表，移除 recommend_score，並格式化 gattered_datetime
        result = [
            {
                "title": item.title,
                "category": item.category,
                "abstract": item.abstract,
                "gattered_datetime": format_date(item.gattered_datetime),
                "link": item.link,
                "data_source": item.data_source,
                "image": item.image
            }
            for log, item in recommendation_logs
        ]

        return jsonify(result), 200

    except Exception as error:
        print(error)
        logging.error(f'[GET ERROR] - {error}')
        abort(500)


@user_history_bp.route('/unrecommend/<uuid:user_uuid>', methods = ['GET'])
def get_unrecommend_items(user_uuid):
    try:
        # # this week
        # today = datetime.today() + timedelta(days=1)
        # # today = datetime(2024, 10, 22) + timedelta(days=1)
        # start_of_week = today - timedelta(days=8)

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

        # 獲取參數中的 data_source
        data_source = request.args.get('data_source', None)
        
        # 查詢條件：user_id, recommend_score，並預過濾 data_source
        unrecommendation_logs_query = DB.session.query(Recommendationlog, Item)\
            .join(Item, Recommendationlog.item_id == Item.uuid)\
            .filter(
                Recommendationlog.user_id == user_uuid,
                Recommendationlog.is_recommend == False,
                Item.gattered_datetime >= start_of_day,
                Item.gattered_datetime <= end_of_day,
            )
        
        # 判斷 data_source 的值
        if data_source == 'news':
            # 如果 data_source 是 'news'，篩選多個來源
            unrecommendation_logs_query = unrecommendation_logs_query.filter(
                Item.data_source.in_(['mit_news', 'cnn_news', 'bbc_news'])
            )
        elif data_source == 'papers':
            # 如果 data_source 是 'papers'，篩選 hf_paper
            unrecommendation_logs_query = unrecommendation_logs_query.filter(Item.data_source == 'hf_paper')
        
        # 排序並限制前十筆
        unrecommendation_logs = unrecommendation_logs_query\
            .order_by(
                Item.gattered_datetime.desc(),
                Item.title.asc()
            )\
            .all()

        # 檢查結果是否為空
        if not unrecommendation_logs:
            return jsonify({'message': 'No unrecommendations found for the specified source'}), 404

        # 構建結果
        result = [{
            'recommend_score': log.recommend_score,
            'item': item.serialize()
        } for log, item in unrecommendation_logs]


        return jsonify(result), 200

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

