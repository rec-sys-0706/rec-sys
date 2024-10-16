import logging
from flask import Flask
from config import Config, DB
from server.models import Item
from server.views import generate_blueprint


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    DB.init_app(app)

    app.register_blueprint(generate_blueprint(Item, 'item'), url_prefix='/api/item')
    # TODO app.register_blueprint(generate_blueprint('app_user'), url_prefix='/api/user')
    # TODO app.register_blueprint(generate_blueprint('behavoir'), url_prefix='/api/behavoir')

    with app.app_context():
        DB.create_all()

    return app
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)