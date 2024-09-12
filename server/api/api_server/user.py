from flask import Blueprint, jsonify, request
from utils import get_db_connection, check_api_key

user_blueprint = Blueprint('users', __name__)


@user_blueprint.route('/', methods=['GET'])
def get_users():
    check_api_key(request)  
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT UUID, Account, Password, Email, Phone FROM dbo.Users')
        rows = cursor.fetchall()
        users_list = []
        for row in rows:
            user = {
                'uuid': row.UUID,
                'account': row.Account,
                'password': row.Password,
                'email': row.Email,
                'phone': row.Phone
            }
            users_list.append(user)
        return jsonify(users_list)
    finally:
        conn.close()


@user_blueprint.route('/<uuid>', methods=['GET'])
def get_user(uuid):
    check_api_key(request) 
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
       
        cursor.execute('SELECT UUID, Account, Password, Email, Phone FROM dbo.Users WHERE UUID = ?', uuid)
        row = cursor.fetchone()
        if row:
            user = {
                'uuid': row.UUID,
                'account': row.Account,
                'password': row.Password,
                'email': row.Email,
                'phone': row.Phone
            }
            return jsonify(user)
        else:
            return jsonify({'error': 'User not found'}), 404
    finally:
        conn.close()