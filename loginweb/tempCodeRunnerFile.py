from flask import Flask, render_template, request, redirect, url_for, session
import pyodbc

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'

# 配置 SQL Server 連接字串
conn_str = (
    "Driver={SQL Server};"
    "Server=LAPTOP-IGBO7T9O\\SQLEXPRESS01;"  # 替換為你的SQL Server名稱
    "Database=data;"  # 替換為你的資料庫名稱
    "Trusted_Connection=yes;"
)

# 連接到資料庫
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        # 使用正確的表名和 schema
        cursor.execute("SELECT * FROM dbo.Users WHERE name=? AND password=?", (name, password))
        user = cursor.fetchone()

        if user:
            session['name'] = name
            return redirect(url_for('welcome'))
        else:
            return '登入失敗，請檢查用戶名和密碼是否正確。'

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        # 插入新用戶到正確的表和 schema
        cursor.execute("INSERT INTO dbo.Users (name, password) VALUES (?, ?)", (name, password))
        conn.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/welcome')
def welcome():
    if 'name' in session:
        return f"歡迎 {session['name']}!"
    else:
        return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
