import pyodbc
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# SQL Server 连接配置
server = 'LAPTOP-IGBO7T9O\SQLEXPRESS01'  # 替换为你的SQL Server服务器名称,1433
database = 'news'  # 替换为你的数据库名称
username = 'admin'  # 替换为你的用户名
password = '1234'  # 替换为你的密码
driver = '{ODBC Driver 17 for SQL Server}'  # 确保你已经安装了相应的ODBC驱动程序

def get_db_connection():
    try:
        conn = pyodbc.connect(
            f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        )
        print("Database connection successful.")
        return conn
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        abort(500, description="Database connection failed")

# API Key 認證
API_KEY = 'api_news_key'

def check_api_key(request):
    auth_header = request.headers.get('Authorization')
    print(f"Authorization Header: {auth_header}")  # 调试信息
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        print(f"Extracted Token: {token}")  # 调试信息
        if token == API_KEY:
            return True
    abort(401, description="Unauthorized")

@app.route('/api/news', methods=['GET', 'POST'])
def get_or_create_news():
    # 檢查API密鑰
    check_api_key(request)
    
    # 獲取所有新聞
    if request.method == 'GET':
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM News')
        rows = cursor.fetchall()

        news_list = []
        for row in rows:
            news = {
                'Id': row[0],
                'Category': row[1],
                'Subcategory': row[2],
                'Title': row[3],
                'Abstract': row[4],
                'Content': row[5],
                'URL': row[6],
                'PublicationDate': row[7]
            }
            news_list.append(news)

        conn.close()
        return jsonify(news_list)

    # 創建新新聞
    elif request.method == 'POST':
        new_news = request.json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        insert_query = '''
        INSERT INTO News (Category, Subcategory, Title, Abstract, Content, URL, PublicationDate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.execute(insert_query, 
                       new_news['Category'], new_news['Subcategory'], new_news['Title'],
                       new_news['Abstract'], new_news['Content'], new_news['URL'], 
                       new_news['PublicationDate'])
        conn.commit()

        conn.close()
        return jsonify(new_news), 201

@app.route('/api/news/<int:news_id>', methods=['GET', 'PUT', 'DELETE'])
def get_update_or_delete_news(news_id):
    # 檢查API密鑰
    check_api_key(request)
    
    conn = get_db_connection()
    cursor = conn.cursor()

    # 獲取特定ID的新聞
    if request.method == 'GET':
        cursor.execute('SELECT * FROM News WHERE Id = ?', (news_id,))
        row = cursor.fetchone()

        if row is None:
            abort(404, description="News not found")

        news = {
            'Id': row[0],
            'Category': row[1],
            'Subcategory': row[2],
            'Title': row[3],
            'Abstract': row[4],
            'Content': row[5],
            'URL': row[6],
            'PublicationDate': row[7]
        }

        conn.close()
        return jsonify(news)

    # 更新新聞
    elif request.method == 'PUT':
        updated_data = request.json
        update_query = '''
        UPDATE News
        SET Category = ?, Subcategory = ?, Title = ?, Abstract = ?, Content = ?, URL = ?, PublicationDate = ?
        WHERE Id = ?
        '''
        cursor.execute(update_query, 
                       updated_data.get('Category'), updated_data.get('Subcategory'), updated_data.get('Title'),
                       updated_data.get('Abstract'), updated_data.get('Content'), updated_data.get('URL'),
                       updated_data.get('PublicationDate'), news_id)
        conn.commit()

        if cursor.rowcount == 0:
            abort(404, description="News not found")

        conn.close()
        return jsonify(updated_data)

    # 刪除新聞
    elif request.method == 'DELETE':
        cursor.execute('DELETE FROM News WHERE Id = ?', (news_id,))
        conn.commit()

        if cursor.rowcount == 0:
            abort(404, description="News not found")

        conn.close()
        return '', 204

@app.route('/api/users', methods=['GET'])
def get_users():
    # 檢查API密鑰
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Users')  # 假設你有一個Users表
    rows = cursor.fetchall()

    users_list = []
    for row in rows:
        user = {
            'UserID': row[0],
            'UserName': row[1],
            'PasswordHash': row[2],
            'LineID': row[3]
        }
        users_list.append(user)

    conn.close()
    return jsonify(users_list)

@app.route('/api/user-news-records', methods=['GET'])
def get_user_news_records():
    # 檢查API密鑰
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM UserNewsRecords')  # 假設你有一個UserNewsRecords表
    rows = cursor.fetchall()

    records_list = []
    for row in rows:
        record = {
            'UserID': row[0],
            'NewsID': row[1],
            'ViewedDate': row[2]
        }
        records_list.append(record)

    conn.close()
    return jsonify(records_list)

if __name__ == '__main__':
    app.run(debug=True)
