from flask import Flask
from .api.bp_generator import generate_blueprint


app = Flask(__name__)
app.register_blueprint(generate_blueprint('item'), url_prefix='/api/item')
app.register_blueprint(generate_blueprint('app_user'), url_prefix='/api/user')
app.register_blueprint(generate_blueprint('behavoir'), url_prefix='/api/behavoir')
