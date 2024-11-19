from flask import Blueprint, jsonify, abort, request
from api.utils import get_db_connection, check_api_key
import re

user_blueprint = Blueprint('user', __name__)

# OK
# Get all users
@user_blueprint.route('/', methods=['GET'])
def get_users():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT UUID, Account, Password, Email, Phone, LineID FROM dbo.[User]')
        rows = cursor.fetchall()
        users_list = []
        for row in rows:
            user = {
                'uuid': row.UUID.lower(),
                'account': row.Account,
                'password': row.Password,
                'email': row.Email,
                'phone': row.Phone,
                'line_id': row.LineID
            }
            users_list.append(user)
        return jsonify(users_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
# Get user by verification account and password
@user_blueprint.route('/verification', methods=['GET'])
def get_user():
    check_api_key(request)
    user_data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT UUID, Account, Password, Email, Phone, LineID FROM dbo.[User] WHERE Account = ? and Password = ?', (user_data['account'], user_data['password']))
        row = cursor.fetchone()
        if row:
            user = {
                'uuid': row.UUID.lower(),
                'account': row.Account,
                'password': row.Password,
                'email': row.Email,
                'phone': row.Phone,
                'line_id': row.LineID,
                'message': 'User found'
            }
            return jsonify(user)
        else:
            return jsonify({'message': 'User not found'}), 404
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        conn.close()

# OK
# Create a new user
@user_blueprint.route('/', methods=['POST'])
def create_user():
    check_api_key(request)
    new_user = request.json
    # Validate required fields
    if not all(k in new_user for k in ('account', 'password', 'email')):
        return jsonify({'error': 'Account, Password, and Email are required'}), 400
    
    # Basic email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", new_user['email']):
        return jsonify({'error': 'Invalid email format'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO dbo.[User] (UUID, Account, Password, Email, Phone, LineID)
            VALUES (NEWID(), ?, ?, ?, ?, ?)
        ''', (new_user['account'], new_user['password'], new_user['email'], new_user.get('phone'), new_user.get('lineid')))  # 用元組替代字典
        
        conn.commit()
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
# Update a user by UUID
@user_blueprint.route('/<uuid>', methods=['PUT'])
def update_user(uuid):
    check_api_key(request)
    updated_user = request.json
    # Validate required fields
    if not all(k in updated_user for k in ('account', 'password', 'email')):
        return jsonify({'error': 'Account, Password, and Email are required'}), 400
    
    # Basic email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", updated_user['email']):
        return jsonify({'error': 'Invalid email format'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE dbo.[User]
            SET Account = ?, Password = ?, Email = ?, Phone = ?, LineID = ?
            WHERE UUID = ?
        ''', (updated_user['account'], updated_user['password'], updated_user['email'], updated_user.get('phone'), updated_user.get('line_id'), uuid))
        
        conn.commit()
        return jsonify({'message': 'User updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# OK
# Delete a user by UUID
@user_blueprint.route('/<uuid>', methods=['DELETE'])
def delete_user(uuid):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM dbo.[User] WHERE UUID = ?', (uuid,))
        conn.commit()
        return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()
