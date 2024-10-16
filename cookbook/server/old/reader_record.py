from flask import Blueprint, jsonify, abort, request
from server.utils import get_db_connection, check_api_key
import datetime

reader_record_blueprint = Blueprint('reader_record', __name__)

# OK
@reader_record_blueprint.route('/', methods=['GET'])
def get_reader_records():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT RecordID, UserUUID, NewsID, ReadDate FROM dbo.ReaderRecord')
        rows = cursor.fetchall()
        records_list = []
        for row in rows:
            record = {
                'record_id': row.RecordID,
                'user_uuid': row.UserUUID,
                'news_id': row.NewsID,
                'read_date': row.ReadDate
            }
            records_list.append(record)
        return jsonify(records_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
@reader_record_blueprint.route('/<int:record_id>', methods=['GET'])
def get_reader_record_by_id(record_id):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT RecordID, UserUUID, NewsID, ReadDate FROM dbo.ReaderRecord WHERE RecordID = ?', record_id)
        row = cursor.fetchone()
        if row:
            record = {
                'record_id': row.RecordID,
                'user_uuid': row.UserUUID,
                'news_id': row.NewsID,
                'read_date': row.ReadDate
            }
            return jsonify(record)
        else:
            return jsonify({'error': 'Record not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
@reader_record_blueprint.route('/', methods=['POST'])
def create_reader_record():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        new_record = request.json
        # Validate required fields
        if not all(k in new_record for k in ('user_uuid', 'news_id', 'read_date')):
            return jsonify({'error': 'UserUUID, NewsID, and ReadDate are required'}), 400

        # Parse read_date
        try:
            new_record['read_date'] = datetime.datetime.strptime(new_record['read_date'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({'error': 'ReadDate format must be YYYY-MM-DD HH:MM:SS'}), 400

        cursor.execute('''
            INSERT INTO dbo.ReaderRecord (UserUUID, NewsID, ReadDate)
            VALUES (?, ?, ?)
        ''', (new_record['user_uuid'], new_record['news_id'], new_record['read_date']))
        conn.commit()
        return jsonify({'message': 'Reader record created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
@reader_record_blueprint.route('/<int:record_id>', methods=['PUT'])
def update_reader_record(record_id):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updated_record = request.json
        # Validate required fields
        if not all(k in updated_record for k in ('user_uuid', 'news_id', 'read_date')):
            return jsonify({'error': 'UserUUID, NewsID, and ReadDate are required'}), 400

        # Parse read_date
        try:
            updated_record['read_date'] = datetime.datetime.strptime(updated_record['read_date'], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({'error': 'ReadDate format must be YYYY-MM-DD HH:MM:SS'}), 400
        
        cursor.execute('''
            UPDATE dbo.ReaderRecord
            SET UserUUID = ?, NewsID = ?, ReadDate = ?
            WHERE RecordID = ?
        ''', (updated_record['user_uuid'], updated_record['news_id'], updated_record['read_date'], record_id))
        conn.commit()
        return jsonify({'message': 'Reader record updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
@reader_record_blueprint.route('/<int:record_id>', methods=['DELETE'])
def delete_reader_record(record_id):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM dbo.ReaderRecord WHERE RecordID = ?', (record_id,))
        conn.commit()
        return jsonify({'message': 'Reader record deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()