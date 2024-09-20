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
# Get user by UUID
@user_blueprint.route('/<uuid>', methods=['GET'])
def get_user(uuid):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT UUID, Account, Password, Email, Phone, LineID FROM dbo.[User] WHERE UUID = ?', uuid)
        row = cursor.fetchone()
        if row:
            user = {
                'uuid': row.UUID.lower(),
                'account': row.Account,
                'password': row.Password,
                'email': row.Email,
                'phone': row.Phone,
                'line_id': row.LineID
            }
            return jsonify(user)
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

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
            INSERT INTO dbo.User (UUID, Account, Password, Email, Phone, LineID)
            VALUES (NEWID(), @account, @password, @email, @phone, @line_id)
        ''', {'account': new_user['account'], 'password': new_user['password'], 'email': new_user['email'], 'phone': new_user.get('phone'), 'line_id': new_user.get('line_id')})
        conn.commit()
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

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
            UPDATE dbo.User
            SET Account = @account, Password = @password, Email = @email, Phone = @phone, LineID = @line_id
            WHERE UUID = @uuid
        ''', {'account': updated_user['account'], 'password': updated_user['password'], 'email': updated_user['email'], 'phone': updated_user.get('phone'), 'line_id': updated_user.get('line_id'), 'uuid': uuid})
        conn.commit()
        return jsonify({'message': 'User updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# Delete a user by UUID
@user_blueprint.route('/<uuid>', methods=['DELETE'])
def delete_user(uuid):
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM dbo.User WHERE UUID = @uuid', {'uuid': uuid})
        conn.commit()
        return jsonify({'message': 'User deleted successfully'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()
