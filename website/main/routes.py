from flask import Blueprint, render_template

main_bp = Blueprint('main',
                    __name__,
                    url_prefix='/main')

@main_bp.route('/')
def home():
    return render_template('index.html')

@main_bp.route('/news/<string:db_name>/<int:news_id>')
def news_article(db_name, news_id):
    # Render static pages in this function, ignoring database queries
    title = f"News Title {news_id}"  # example
    content = "This is a static news article content."  # static content
    return render_template('./website/main/news.html', title=title, content=content)
