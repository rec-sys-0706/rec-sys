import logging
import uuid

from flask import Blueprint, request, jsonify, abort
from flask_jwt_extended import create_access_token, jwt_required
from werkzeug.security import generate_password_hash, check_password_hash
from server.utils import validate_dict_keys

from server.models.user import User, UserSchema
from server.schemas import UserSchema
from config import DB

user_blueprint = Blueprint('user', __name__)

# 初始化 Marshmallow schema
user_schema = UserSchema()
users_schema = UserSchema(many=True)  # 用來序列化多個用戶

# 取得所有用戶
@user_blueprint.route('', methods=['GET'])
@jwt_required
def get_users():
    users = User.query.all()  # 從數據庫中查詢所有用戶
    if not users:
        return jsonify({"message": "No users found"}), 404
    result = users_schema.dump(users)  # 序列化數據
    return jsonify(result), 200

# 根據 UUID 查詢特定用戶
@user_blueprint.route('/<uuid:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"message": "User not found"}), 404
    result = user_schema.dump(user)
    return jsonify(result), 200

# 註冊，創建新用戶    
@user_blueprint.route('/register', methods=['POST'])
@jwt_required
def register():
    data = request.json
    account = data.get('account')
    password = data.get('password')
    email = data.get('email')
    line_id = data.get('line_id', None)

    if not account or not password or not email:
        return jsonify({"msg": "Missing username, email, or password"}), 400

    # 檢查是否已存在用戶
    if User.query.filter_by(account=account).first():
        return jsonify({"msg": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "Email already registered"}), 400

    # give user an uuid
    id = str(uuid.uuid4())
    # 創建新用戶
    password_hash = generate_password_hash(password)
    logging.info(len(password_hash))
    new_user = User(uuid=id, account=account, password=password_hash, email=email, line_id=line_id)
    DB.session.add(new_user)
    DB.session.commit()

# 登入、驗證
@user_blueprint.route('/login', methods=['POST'])
@jwt_required
def login():
    data = request.json
    account = data.get('account')
    password = data.get('password')

    if not account or not password:
        return jsonify({"msg": "Missing username or password"}), 400

    user = User.query.filter_by(account=account).first()

    # 驗證用戶名和密碼
    if not user or not check_password_hash(user.password, password):
        return jsonify({"msg": "Invalid credentials"}), 401

    # 登錄成功，生成 JWT
    access_token = create_access_token(identity=account)
    return jsonify({"access_token": access_token}), 200

# 更新用戶資料
@user_blueprint.route('/<uuid:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = user_schema.load(request.json, partial=True)
    headers = [column.name for column in User.__table__.columns]
    # 驗證請求數據的鍵是否有效
    if not validate_dict_keys(data, headers):
        abort(400, description="Invalid keys in the request.")

    try:
        # 將請求中的數據更新到用戶模型中
        for key, value in data.items():
            setattr(user, key, value)

        # 提交更改到數據庫
        DB.session.commit()
        return jsonify({'message': 'Success'}), 204  # 成功修改，返回狀態碼 204
    except Exception as error:
        logging.error(f'UPDATE ERROR: {error}')
        DB.session.rollback()
        abort(500, description="An error occurred during the update.")

# 刪除用戶
@user_blueprint.route('/<uuid:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"message": "User not found"}), 404
    
    try:
        DB.session.delete(user)
        DB.session.commit()
        return jsonify({"message": "User deleted successfully"}), 204  # 204 No Content
    except Exception as e:
        DB.session.rollback()
        return jsonify({"message": f"Error deleting user: {str(e)}"}), 500
