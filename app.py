import logging
from apiflask import APIFlask
from config import Config, DB
from server.bot import linebot_bp
from flask_jwt_extended import JWTManager

from server.views.user_views import user_blueprint
from server.views.item_views import item_blueprint
from server.views.behavior_views import behavior_blueprint
from server.services.browsing_history import user_history_bp
from server.views.recommendation_log_views import recommendation_bp
from server.views.mind_views import mind_blueprint

def create_app():
    app = APIFlask(__name__)
    app.config.from_object(Config)
    app.config['SPEC_FORMAT'] = 'yaml'
    
    # 初始化數據資料庫
    DB.init_app(app)

    jwt = JWTManager(app)

    # TODO history
    app.register_blueprint(linebot_bp, url_prefix='/api/callback')

    app.register_blueprint(user_blueprint, url_prefix='/api/user')
    app.register_blueprint(item_blueprint, url_prefix='/api/item')
    app.register_blueprint(behavior_blueprint, url_prefix='/api/behavior')
    app.register_blueprint(recommendation_bp, url_prefix='/api/recommend')
    app.register_blueprint(mind_blueprint, url_prefix='/api/mind')

    app.register_blueprint(user_history_bp, url_prefix='/api/user_history')

    with app.app_context():
        DB.create_all()

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    app = create_app()
    app.run(host='0.0.0.0', port=80, debug=False)