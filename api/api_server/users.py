from flask import Flask, jsonify, request, abort
import pyodbc
import uuid

app = Flask(__name__)

# SQL Server 連接配置
server = ''
database = ''
username = ''  
password = ''  
driver = '{}'
# 獲取SQL Server資料庫連接
def get_db_connection():
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'
    conn = pyodbc.connect(conn_str)
    return conn

# 暫時不檢查API密鑰
def check_api_key(req):
    pass  # 在測試期間不檢查API密鑰

# 獲取使用者資料的API端點
@app.route('/api/users', methods=['GET'])
def get_users():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 選取使用者表中的所有欄位
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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port='5000')
