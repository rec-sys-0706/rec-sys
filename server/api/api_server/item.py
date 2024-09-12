from flask import Blueprint, jsonify, request
from utils import get_db_connection, check_api_key

news_blueprint = Blueprint('news', __name__)

@news_blueprint.route('/', methods=['GET'])
def get_news():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT Title, Date, Category, Abstract FROM dbo.News')
        rows = cursor.fetchall()
        news_list = []
        for row in rows:
            news_item = {
                'title': row.Title,
                'date': row.Date,
                'category': row.Category,
                'abstract': row.Abstract
            }
            news_list.append(news_item)
        return jsonify(news_list)
    finally:
        conn.close()

@news_blueprint.route('/<title>', methods=['GET'])
def get_news_by_title(title):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT Title, Date, Category, Abstract FROM dbo.News WHERE Title = ?', title)
        row = cursor.fetchone()
        if row:
            news_item = {
                'title': row.Title,
                'date': row.Date,
                'category': row.Category,
                'abstract': row.Abstract
            }
            return jsonify(news_item)
        else:
            return jsonify({'error': 'News item not found'}), 404
    finally:
        conn.close()
