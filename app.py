from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # 在此函数中渲染静态页面，忽略数据库查询
    title = f"News Title {news_id}"  # 只是演示，实际可以根据需要设定标题
    content = "This is a static news article content."  # 静态内容
    return render_template('news.html', title=title, content=content)

if __name__ == '__main__':
    app.run(debug=True)
