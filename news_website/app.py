from flask import Flask, render_template, send_file, request, session
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)
app.secret_key = 'wqF1rXGsOgY8NyKslxTZXE8YFqnbv0FG'

# 讀取news檔案
news_path = './news_website/data/news.json'

with open(news_path, encoding='utf-8') as file:
    article = json.load(file)

# 讀取result檔案
result_path = './news_website/data/result.json'

with open(result_path, encoding='utf-8') as file:
    result = json.load(file)

# 將Category按照字母順序排列
article = sorted(article, key = lambda i: i['Category'])

@app.route('/')
def index():
    for index, item in enumerate(result, start=1):
        item["NO"] = str(index)

    return render_template('backstage.html', users = result)


@app.route('/user')
def user():
    # 在result檔案裡找到指定的user_id
    user_id = request.args.get('user_id')

    session['username'] = user_id

    user_data = [item for item in result if item['user_id'] == user_id][0]

    clicked_news = json.loads(user_data['clicked_news'].replace("'", '"'))

    candidate_news = json.loads(user_data['candidate_news'].replace("'", '"'))

    clicked_times = json.loads(user_data['clicked'].replace("'", '"'))

    # 根據clicked_news去尋找news檔案裡的ID
    clicked_articles = [article for article in article if article['ID'] in clicked_news]

    # 根據candidate_news去尋找news檔案裡的ID
    candidate_articles = [article for article in article if article['ID'] in candidate_news]

    #判斷是否有點擊，點擊加顏色
    for text, time in zip(candidate_articles, clicked_times):
        if(time == 0):
            text['color'] = 'nocolor'
        else:
            text['color'] = 'color'

    # 添加NO標籤
    for index, item in enumerate(clicked_articles, start = 1):
        item["NO"] = str(index)

    return render_template('user_profile.html',user = user_id, clicked_articles = clicked_articles, candidate_articles = candidate_articles)

@app.route('/wordcloud.png')
def wordcloud_image():
    img = io.BytesIO()

    # 在result檔案裡找到指定的user_id
    user_id =  session.get('username', None)

    user_data = [item for item in result if item['user_id'] == user_id][0]

    clicked_news = json.loads(user_data['clicked_news'].replace("'", '"'))

    clicked_articles = [article for article in article if article['ID'] in clicked_news]

    categories =  [article['Title'] for article in clicked_articles]
    wordcloud_text = ' '.join(categories)
    mask = np.array(Image.open("./news_website/static/mask.png"))
    wordcloud = WordCloud(width=800, height=250, background_color='white', mask=mask, contour_color='white', contour_width=1).generate(wordcloud_text)
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
