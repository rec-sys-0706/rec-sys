from flask import Flask, render_template

app = Flask(__name__)

# Sample news data
articles = [
    {
        'title': 'Breaking News: Market Hits All-Time High',
        'content': 'The stock market has reached an all-time high today, with major indices showing significant gains.',
        'author': 'John Doe',
        'date': '2024-08-09'
    },
    {
        'title': 'Tech Innovations: AI Revolutionizing Industries',
        'content': 'Artificial Intelligence is transforming the way businesses operate, from automation to customer service.',
        'author': 'Jane Smith',
        'date': '2024-08-08'
    },
    # Add more news articles here
]

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
