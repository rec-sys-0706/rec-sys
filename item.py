from flask import Flask, jsonify, request, abort
import pyodbc

app = Flask(__name__)

# SQL Server 連接配置
server = 'DESKTOP-KTKE5PO,1433'
database = 'news'
username = 'admin'  
password = 'FJU0922'  
driver = '{ODBC Driver 17 for SQL Server}'

# 獲取SQL Server資料庫連接
def get_db_connection():
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'
    conn = pyodbc.connect(conn_str)
    return conn

# 暫時不檢查API密鑰
def check_api_key(req):
    pass  # 在測試期間不檢查API密鑰

# 獲取新聞資料的API端點
@app.route('/api/news', methods=['GET'])
def get_news():
    check_api_key(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 選取新聞表中的所有欄位
        cursor.execute('SELECT Title, Date, Category,  abstract FROM dbo.News')
        rows = cursor.fetchall()
        news_list = []
        for row in rows:
            news_item = {
                'title': row.Title,
                'date': row.Date,
                'category': row.Category,
                'abstract': row. abstract
            }
            news_list.append(news_item)
        return jsonify(news_list)
    finally:
        conn.close()

if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0', port='5000')
