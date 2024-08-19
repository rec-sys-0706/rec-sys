from flask import Flask, render_template
import pyodbc
from datetime import datetime

app = Flask(__name__, template_folder='C:/Users/user/Desktop/website/.venv/templates', static_folder='C:/Users/user/Desktop/website/.venv/static')

# SQL Server connection string
conn_str = (
    "Driver={SQL Server};"
    "Server=LAPTOP-IGBO7T9O\\SQLEXPRESS01;"
    "Database=cnn;"
    "Trusted_Connection=yes;"
)

@app.route('/')
def index():
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT Category, Subcategory, Title, Abstract, URL, PublicationDate FROM News")
    articles = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
