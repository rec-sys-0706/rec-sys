from flask import Flask
from user import user_blueprint
from item import news_blueprint

app = Flask(__name__)

app.register_blueprint(user_blueprint, url_prefix='/api/users')
app.register_blueprint(news_blueprint, url_prefix='/api/news')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
