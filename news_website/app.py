from flask import Flask, render_template, send_file, request, session
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import io
import json
import csv

app = Flask(__name__)
app.secret_key = 'wqF1rXGsOgY8NyKslxTZXE8YFqnbv0FG'

news_path = './news_website/data/news.tsv'

with open(news_path, encoding = 'utf-8') as csvfile:
    news_file = csv.reader(csvfile, delimiter = "\t")
    # 根據類別開頭字母排序
    news_article = sorted(news_file, key = lambda row: row[1])

result_path = './news_website/data/result.csv'

with open(result_path, newline = '') as csvfile:
    result_file = list(csv.reader(csvfile))
    
    # 添加順序
    for index, item in enumerate(result_file, start = 0):
        item.append(index)

#刪除數據中的第一列
del result_file[0]

@app.route('/')
def index():
    return render_template('backstage.html', users = result_file)


@app.route('/user')
def user():
    # 獲取點擊id
    user_id = request.args.get('user_id')
    # 將id保存到session中
    session['username'] = user_id
    # 根據獲取的id從找到對應的數據
    user_data = [data for data in result_file if user_id in data][0]

    clicked_news = json.loads(user_data[1].replace("'", '"'))

    candidate_news = json.loads(user_data[2].replace("'", '"'))
    # user clicked_news 的新聞資訊
    clicked_articles = [x for x in news_article for y in clicked_news if x[0] == y]
    # 找到user candidate_news 的新聞資訊
    candidate_articles = [x for x in news_article for y in candidate_news if x[0] == y]

    # 判定user是否有點擊(添加class的功能)
    num = 0
    for time in user_data[3]:
        if time == '1' or time == '0':
            if(time == '0'):
                candidate_articles[num].append("nocolor")
            else:
                candidate_articles[num].append("color") # 若點擊則上色
            num = num + 1

    #添加編號
    for index, item in enumerate(clicked_articles, start = 1):
        item.append(index)

    return render_template('user_profile.html',user = user_id, clicked_articles = clicked_articles, candidate_articles = candidate_articles)

# 文字雲
@app.route('/wordcloud.png')
def wordcloud_image():
    img = io.BytesIO()

    # 從session中獲取點擊ID
    user_id =  session.get('username', None)

    user_data = [data for data in result_file if user_id in data][0]
    
    clicked_news = json.loads(user_data[1].replace("'", '"'))

    clicked_articles = [x for x in news_article for y in clicked_news if x[0] == y]

    titles = []
    # 將文章標題存進list裡面
    for title in clicked_articles:
        titles.append(title[3])
    
    # 將所有標題合併成一個字串
    wordcloud_text = ' '.join(titles)
    mask = np.array(Image.open("./news_website/static/mask.png"))
    wordcloud = WordCloud(width=800, height=250, background_color='white', mask=mask, contour_color='white', contour_width=1).generate(wordcloud_text)
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
