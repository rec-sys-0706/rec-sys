import logging
from apiflask import APIFlask
from flask import Flask
from apiflask import APIFlask
from config import Config, DB
from server.models import init_db_models
from server.models.api import BehaviorIn, ItemIn, UserIn
from server.models.item import Item, ItemSchema
from server.models.user import User, UserSchema
from server.models.behavior import Behavior, BehaviorSchema
from server.views import generate_blueprint
from server.bot import linebot_bp
from server.views.auth import auth
from flask_jwt_extended import JWTManager
import os

from server.views.user_views import user_blueprint


def create_app():
    app = APIFlask(__name__)
    app.config.from_object(Config)
    app.config['SPEC_FORMAT'] = 'yaml'
    
    # 初始化數據資料庫
    DB.init_app(app)

    jwt = JWTManager()
    jwt.init_app(app)

    # item_schema = ItemSchema()
    # user_schema = UserSchema()
    # behavior_schema = BehaviorSchema()
    # app.register_blueprint(generate_blueprint(Item, item_schema, 'item', ItemIn), url_prefix='/api/item')
    # app.register_blueprint(generate_blueprint(User, user_schema, 'app_user', UserIn), url_prefix='/api/user')
    # app.register_blueprint(generate_blueprint(Behavior, behavior_schema, 'behavoir', BehaviorIn), url_prefix='/api/behavoir')
    # app.register_blueprint(auth, url_prefix = '/api/auth')
    # TODO history
    app.register_blueprint(linebot_bp, url_prefix='/api/callback')

    app.register_blueprint(user_blueprint, url_prefix='/api/user')

    with app.app_context():
        DB.create_all()

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)