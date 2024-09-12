from flask import Flask
from user import user_blueprint
from item import item_blueprint
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)


app.register_blueprint(user_blueprint, url_prefix='/api/users')
app.register_blueprint(item_blueprint, url_prefix='/api/news')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)