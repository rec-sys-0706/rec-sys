from datetime import datetime
import logging
import uuid

from apiflask import APIBlueprint
from flask import request, jsonify, abort
from flask_jwt_extended import create_access_token, jwt_required
from werkzeug.security import generate_password_hash, check_password_hash
from server.utils import generate_random_scores, validate_dict_keys

from server.models.user import User, UserSchema
from server.models.item import Item
from server.models.recommendation_log import Recommendationlog
from config import DB

user_blueprint = APIBlueprint('app_user', __name__)

user_schema = UserSchema()

# get_all_users
@user_blueprint.route('', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify({
        'data': [
            {
                'uuid': user.uuid,
                'account': user.account,
                'email': user.email,
            } for user in users
        ]
    }), 200

# get_certain_user_uuid
@user_blueprint.route('/<uuid:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'data': user.serialize()
    }), 200

# register    
@user_blueprint.route('/register', methods=['POST'])
def user_register():
    data = request.json
    account = data.get('account')
    password = data.get('password')
    email = data.get('email')
    line_id = data.get('line_id', None)

    if not account or not password or not email:
        return jsonify({"msg": "Missing username, email, or password"}), 400

    # check exist or not
    if User.query.filter_by(account=account).first():
        return jsonify({"msg": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "Email already registered"}), 400

    # give user an uuid
    id = str(uuid.uuid4())
    
    password_hash = generate_password_hash(password)
    logging.info(len(password_hash))
    new_user = User(uuid=id, account=account, password=password_hash, email=email, line_id=line_id)
    DB.session.add(new_user)
    DB.session.commit()
    # return jsonify({"msg": "Success Register"}), 200

    items = Item.query.all()
    item_list = [{'uuid': str(item.uuid), 'title': item.title, 'abstract': item.abstract, 'link': item.link, 'data_source':item.data_source, 'gattered_datetime':item.gattered_datetime} for item in items] 

    user_data = {
        'uuid': id,
        'account': account,
        'email': email,
        'line_id': line_id
    }

    recommendations = generate_random_scores(item_list, [user_data])

    try:
        for recommendation in recommendations:
            new_recommendation = Recommendationlog(
                uuid=recommendation['uuid'],
                user_id=recommendation['user_id'],
                item_id=recommendation['item_id'],
                recommend_score=recommendation['recommend_score'],
                gattered_datetime=recommendation['gattered_datetime']
            )
            DB.session.add(new_recommendation)

        # 提交所有更改user_uuid
        DB.session.commit()
        logging.info("Recommendation logs created successfully.")
    except Exception as e:
        logging.error(f"Error saving recommendation logs: {e}")
        DB.session.rollback()
        return jsonify({"msg": "Error saving recommendation logs"}), 500

    return jsonify({"msg": "Success Register and Recommendations Generated"}), 200
    

# login, verify
@user_blueprint.route('/login', methods=['POST'])
def login():
    data = request.json
    account = data.get('account')
    password = data.get('password')

    if not account or not password:
        return jsonify({"msg": "Missing username or password"}), 400

    user = User.query.filter_by(account=account).first()

    # ckeck account and password
    if not user or not check_password_hash(user.password, password):
        return jsonify({"msg": "Invalid credentials"}), 401

    # JWT token
    access_token = create_access_token(identity=user.uuid)
    return jsonify({"access_token": access_token, "uuid":user.uuid}), 200

# update_user_data
@user_blueprint.route('/<uuid:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = user_schema.load(request.json, partial=True)
    headers = [column.name for column in User.__table__.columns]

    if not validate_dict_keys(data, headers):
        abort(400, description="Invalid keys in the request.")

    try:

        for key, value in data.items():
            setattr(user, key, value)

        DB.session.commit()
        return jsonify({'message': 'Success'}), 204  # Success 204
    except Exception as error:
        logging.error(f'UPDATE ERROR: {error}')
        DB.session.rollback()
        abort(500, description="An error occurred during the update.")

# delete_user
@user_blueprint.route('/<uuid:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"message": "User not found"}), 404
    
    try:
        DB.session.delete(user)
        DB.session.commit()
        return jsonify({"message": "User and related data deleted successfully"}), 204  # 204 no content
    except Exception as e:
        DB.session.rollback()
        return jsonify({"message": f"Error deleting user: {str(e)}"}), 500
    

@user_blueprint.route('/mind', methods=['POST'])
def create_user_data():
    data = request.json
    account = data.get('account')
    password = data.get('password')
    email = data.get('email')
    line_id = data.get('line_id', None)

    # 檢查必要欄位
    if not account or not password or not email:
        return jsonify({"msg": "Missing username, email, or password"}), 400

    # 檢查帳號和電子郵件是否已存在
    if User.query.filter_by(account=account).first():
        return jsonify({"msg": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "Email already registered"}), 400

    # 生成 UUID 和密碼哈希
    id = str(uuid.uuid4())
    password_hash = generate_password_hash(password)

    # 新增使用者到資料庫
    new_user = User(uuid=id, account=account, password=password_hash, email=email, line_id=line_id)
    DB.session.add(new_user)
    DB.session.commit()

    return jsonify({"msg": "Success Register"}), 200  

# 透過 account 獲取 user_uuid
@user_blueprint.route('/get_uuid_by_account', methods=['GET'])
def get_user_uuid_by_account():
    account = request.args.get('account')
    
    if not account:
        return jsonify({"msg": "Missing account parameter"}), 400

    # 查詢 User 表中的 account
    user = User.query.filter_by(account=account).first()
    if not user:
        return jsonify({"msg": "User not found"}), 404

    # 返回 user_uuid
    return jsonify({
        'user_uuid': str(user.uuid)
    }), 200 
