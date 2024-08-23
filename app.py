from flask import Flask, render_template

app = Flask(__name__, template_folder='C:/Users/user/Desktop/website/.venv/templates', static_folder='C:/Users/user/Desktop/website/.venv/static')

# SQL Server connection string
conn_str = (
    "Driver={SQL Server};"
    "Server=LAPTOP-IGBO7T9O\\SQLEXPRESS01;"
    "Database=cnn;"
    "Trusted_Connection=yes;"
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # sstatic content
    return render_template('news.html', title=title, content=content)

def news():
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT Category, Subcategory, Title, Abstract, URL, PublicationDate FROM News")
    articles = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('news.html', articles=articles)


if __name__ == '__main__':
    app.run(debug=True)
