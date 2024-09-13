import urllib.parse
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# 初始化 Flask 應用程式與資料庫
def create_app(SQLSERVER, SQL_DATABASE, SQL_USERNAME,SQL_PASSWORD):
    app = Flask(__name__)
    params = urllib.parse.quote_plus(
        f'Driver={{ODBC Driver 17 for SQL Server}};Server={SQLSERVER};Database={SQL_DATABASE};UID={SQL_USERNAME};PWD={SQL_PASSWORD};Trusted_Connection=yes;'
    )
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mssql+pyodbc:///?odbc_connect={params}'
    return app

# 初始化資料庫
def init_db(app):
    db = SQLAlchemy(app)
    return db

# 定義 Stories Model
def define_stories_model(db):
    class Stories(db.Model):
        __tablename__ = 'stories'
        id = db.Column(db.INTEGER, primary_key=True)
        category = db.Column(db.String('max'), unique=False, nullable=False)
        subcategory = db.Column(db.String('max'), unique=False, nullable=False)
        title = db.Column(db.String('max'), unique=False, nullable=False)
        date = db.Column(db.String('max'), unique=False, nullable=False)
        abstract = db.Column(db.String('max'), unique=False, nullable=False)
        contents = db.Column(db.String('max'), unique=False, nullable=False)
        url = db.Column(db.String('max'), unique=False, nullable=False)

        def __init__(self, category, subcategory, title, date, abstract, contents, url):
            self.category = category
            self.subcategory = subcategory
            self.title = title
            self.date = date
            self.abstract = abstract
            self.contents = contents
            self.url = url

    return Stories

# 建立資料庫
def setup_database(app, db):
    with app.app_context():
        db.drop_all()
        db.create_all()

if __name__ == '__main__':
    app = create_app("SQLSERVER", "SQL_DATABASE", "SQL_USERNAME","SQL_PASSWORD")
    db = init_db(app)
    
    Stories = define_stories_model(db)
    
    setup_database(app, db)