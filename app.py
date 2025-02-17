from datetime import timedelta
import logging
import uuid
from apiflask import APIFlask
from config import Config, DB
#from server.bot import linebot_bp
from flask_jwt_extended import JWTManager
from flask_compress import Compress 

from server.views.user_views import user_blueprint
from server.views.item_views import item_blueprint
from server.views.behavior_views import behavior_blueprint
from server.services.browsing_history import user_history_bp
from server.views.recommendation_log_views import recommendation_bp
from server.views.mind_views import mind_blueprint

def create_app(website_only=True):
    app = APIFlask(__name__)
    app.config.from_object(Config)
    app.config['SPEC_FORMAT'] = 'yaml'
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=20)


    if not website_only:
        # Server settings
        DB.init_app(app)

        # 初始化 Flask-Compress
        Compress(app)
        
        jwt = JWTManager(app)

        # TODO history
        #app.register_blueprint(linebot_bp, url_prefix='/api/callback')

        app.register_blueprint(user_blueprint, url_prefix='/api/user')
        app.register_blueprint(item_blueprint, url_prefix='/api/item')
        app.register_blueprint(behavior_blueprint, url_prefix='/api/behavior')
        app.register_blueprint(recommendation_bp, url_prefix='/api/recommend')
        app.register_blueprint(mind_blueprint, url_prefix='/api/mind')

        app.register_blueprint(user_history_bp, url_prefix='/api/user_history')

        with app.app_context():
            DB.create_all()

    from website.main.routes import main_bp
    app.secret_key = uuid.uuid4().hex
    app.register_blueprint(main_bp)

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)