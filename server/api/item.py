from flask import Blueprint, jsonify, abort, request
from api.utils import get_db_connection, check_api_key
import datetime
news_blueprint = Blueprint('news', __name__)

# OK
@news_blueprint.route('/', methods=['GET'])
def get_news():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT NewsID, Title, Date, Category, Abstract FROM dbo.News')
        rows = cursor.fetchall()
        news_list = []
        for row in rows:
            news_item = {
                'news_id': row.NewsID,
                'title': row.Title,
                'date': row.Date,
                'category': row.Category,
                'abstract': row.Abstract
            }
            news_list.append(news_item)
        return jsonify(news_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
@news_blueprint.route('/<int:news_id>', methods=['GET'])
def get_news_by_id(news_id):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT NewsID, Title, Date, Category, Abstract FROM dbo.News WHERE NewsID = ?', news_id)
        row = cursor.fetchone()
        if row:
            news_item = {
                'news_id': row.NewsID,
                'title': row.Title,
                'date': row.Date,
                'category': row.Category,
                'abstract': row.Abstract
            }
            return jsonify(news_item)
        else:
            return jsonify({'error': 'News item not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@news_blueprint.route('/', methods=['POST'])
def create_news():
    check_api_key(request)
    try:
        new_news = request.json
        if not all(k in new_news for k in ('title', 'date')):
            return jsonify({'error': 'Title and Date are required'}), 400

        try:
            new_news['date'] = datetime.datetime.strptime(new_news['date'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({'error': 'Date format must be YYYY-MM-DD HH:MM:SS'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO dbo.News (Title, Date, Category, Abstract)
            VALUES (?, ?, ?, ?)
        ''', (new_news['title'], new_news.get('date'), new_news.get('category'), new_news.get('abstract')))
        conn.commit()
        return jsonify({'message': 'News created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@news_blueprint.route('/<int:news_id>', methods=['PUT'])
def update_news(news_id):
    check_api_key(request)
    try:
        updated_news = request.json
        # Validate required fields
        if not all(k in updated_news for k in ('title', 'date')):
            return jsonify({'error': 'Title and Date are required'}), 400

        # Parse date
        try:
            updated_news['date'] = datetime.datetime.strptime(updated_news['date'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({'error': 'Date format must be YYYY-MM-DD HH:MM:SS'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE dbo.News
            SET Title = ?, Date = ?, Category = ?, Abstract = ?
            WHERE NewsID = ?
        ''', (updated_news['title'], updated_news['date'], updated_news.get('category'), updated_news.get('abstract'), news_id))
        conn.commit()
        return jsonify({'message': 'News updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@news_blueprint.route('/<int:news_id>', methods=['DELETE'])
def delete_news(news_id):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM dbo.News WHERE NewsID = ?', news_id)
        conn.commit()
        return jsonify({'message': 'News deleted successfully'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()
