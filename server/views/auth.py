import logging
import uuid
from apiflask import APIBlueprint
from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager,create_access_token
from server.models.user import User
from config import DB


auth = APIBlueprint('auth', __name__)

@auth.route('/register', methods=['POST'])
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

    id = str(uuid.uuid4())
    # 創建新用戶
    password_hash = generate_password_hash(password)
    logging.info(len(password_hash))
    new_user = User(uuid=id, account=account, password=password_hash, email=email, line_id=line_id)
    DB.session.add(new_user)
    DB.session.commit()

    return jsonify({"msg": "User registered successfully"}), 201
    
@auth.route('/login', methods=['POST'])
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

