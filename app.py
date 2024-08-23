from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # sstatic content
    return render_template('news.html', title=title, content=content)

if __name__ == '__main__':
    app.run(debug=True)
