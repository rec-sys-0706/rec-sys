import logging
import uuid

from apiflask import APIBlueprint
from flask import request, jsonify, abort
from server.utils import validate_dict_keys

from server.models.recommendation_log import Recommendationlog, RecommendationlogSchema
from config import DB

recommendation_bp = APIBlueprint('recommendation_log', __name__)

recommendation_schema = RecommendationlogSchema()

headers = [column.name for column in Recommendationlog.__table__.columns]

@recommendation_bp.route('/model', methods = ['Post'])
def create_recommend_items():
        input = request.json
        # input['uuid'] = str(uuid.uuid4())
        input['clicked'] = input.get('clicked', 0)
        if not validate_dict_keys(input, headers):
            abort(400)
        try:
            item = Recommendationlog(**input)
            DB.session.add(item)
            DB.session.commit()
            return jsonify({'message': 'Success'}), 201
        except Exception as error:
            logging.error(f'[INSERT ERROR] - {error}')
            DB.session.rollback()
            abort(500) 

@recommendation_bp.route('/<uuid:recommend_uuid>', methods=['PUT'])
def update_recommendation_log(recommend_uuid):
    try:
        # get clikced_record
        input_data = request.get_json()
        clicked = input_data.get('clicked', False)

        # search for recommendation_log_uuid
        recommendation_log = Recommendationlog.query.filter_by(uuid=recommend_uuid).first()

        if not recommendation_log:
            return jsonify({'message': 'Recommendation log not found'}), 404

        # update
        recommendation_log.clicked = clicked
        DB.session.commit()

        return jsonify({'message': 'Recommendation log updated successfully'}), 200

    except Exception as error:

        logging.error(f'[UPDATE ERROR] - {error}')
        DB.session.rollback()
        return jsonify({'message': 'Failed to update recommendation log'}), 500
    



