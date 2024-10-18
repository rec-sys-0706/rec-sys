import logging
from apiflask import APIFlask
from flask import Flask
from config import Config, DB
from server.models.api import BehaviorIn, ItemIn, UserIn
from server.models.item import Item, ItemSchema
from server.models.user import User, UserSchema
from server.models.behavior import Behavior, BehaviorSchema
from server.views import generate_blueprint
# from server.bot import linebot_bp
from server.views.verify import auth

def create_app():
    app = APIFlask(__name__)
    app.config.from_object(Config)
    app.config['SPEC_FORMAT'] = 'yaml'
    
    # DB.init_app(app)


    item_schema = ItemSchema()
    user_schema = UserSchema()
    behavior_schema = BehaviorSchema()
    app.register_blueprint(generate_blueprint(Item, item_schema, 'item', ItemIn), url_prefix='/api/item')
    app.register_blueprint(generate_blueprint(User, user_schema, 'app_user', UserIn), url_prefix='/api/user')
    app.register_blueprint(generate_blueprint(Behavior, behavior_schema, 'behavoir', BehaviorIn), url_prefix='/api/behavoir')
    app.register_blueprint(auth, url_prefix = '/api/auth')
    # TODO history
    # app.register_blueprint(linebot_bp, url_prefix='/api/callback')

    # with app.app_context():
    #     DB.create_all()

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)


# from flask import Flask, jsonify, request
# import json
# import os

# app = Flask(__name__)

# # # 定義 JSON 檔案的路徑，使用 os.path.join 確保跨平台的路徑正確性
# # DATA_FILE = os.path.join(os.path.dirname(__file__), 'data.json')

# # 定義讀取和寫入 JSON 檔案的函數
# def read_json():
#     with open('data.json', 'r') as f:
#         return json.load(f)

# def write_json(data):
#     with open('data.json', 'w') as f:
#         json.dump(data, f, indent=4)

# # 取得所有使用者
# @app.route('/users', methods=['GET'])
# def get_users():
#     data = read_json()
#     return jsonify(data['users'])

# # 新增一個使用者
# @app.route('/users', methods=['POST'])
# def add_user():
#     data = read_json()
    
#     # 從請求中獲取數據
#     new_user = {
#         "uuid": request.json['uuid'],
#         "account": request.json['account'],
#         "password": request.json['password'],
#         "email": request.json['email'],
#         "lineid": request.json['lineid']
#     }

#     # 新增到使用者列表
#     data['users'].append(new_user)
    
#     # 寫回 JSON 檔案
#     write_json(data)
    
#     return jsonify(new_user), 201

# # 更新一個使用者的資訊
# @app.route('/users/<uuid>', methods=['PUT'])
# def update_user(uuid):
#     data = read_json()
    
#     for user in data['users']:
#         if user['uuid'] == uuid:
#             user['account'] = request.json['account']
#             user['password'] = request.json['password']
#             user['email'] = request.json['email']
#             user['lineid'] = request.json['lineid']
            
#             write_json(data)
#             return jsonify(user)
    
#     return jsonify({'message': 'User not found'}), 404

# # 刪除一個使用者
# @app.route('/users/<uuid>', methods=['DELETE'])
# def delete_user(uuid):
#     data = read_json()
    
#     new_users = [user for user in data['users'] if user['uuid'] != uuid]
    
#     if len(new_users) == len(data['users']):
#         return jsonify({'message': 'User not found'}), 404
    
#     data['users'] = new_users
#     write_json(data)
    
#     return jsonify({'message': 'User deleted'})

# if __name__ == '__main__':
#     app.run(debug=True)