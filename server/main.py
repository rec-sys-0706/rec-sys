from flask import Flask
from api.user import user_blueprint
from api.item import news_blueprint
from api.reader_record import reader_record_blueprint 

app = Flask(__name__)

app.register_blueprint(user_blueprint, url_prefix='/api/user')
app.register_blueprint(news_blueprint, url_prefix='/api/news')
app.register_blueprint(reader_record_blueprint, url_prefix='/api/reader_record')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
