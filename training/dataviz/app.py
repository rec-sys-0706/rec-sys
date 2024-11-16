from flask import Flask, render_template, send_file, request, session
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import io
import pandas as pd
import ast
from utils import reclassify_category

app = Flask(__name__)
app.secret_key = 'wqF1rXGsOgY8NyKslxTZXE8YFqnbv0FG'

news = pd.read_csv('./training/dataviz/news.tsv',
                    sep='\t',
                    names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                    index_col='news_id')
news['category'] = news.apply(reclassify_category, axis=1)
result = pd.read_csv('./training/dataviz/eval_result_v2.csv', index_col='user_id').sort_index()

@app.route('/')
def index():
    return render_template('user_list.html', users = result)

@app.route('/user')
def user():
    # 獲取點擊id
    user_id = request.args.get('user_id')
    # 將id保存到session中
    session['username'] = user_id
    # 根據獲取的id從找到clicked_news的新聞資訊
    clicked_news_id = ast.literal_eval(result.loc[user_id]['clicked_news'])
    clicked_news = news.loc[clicked_news_id]
    # 按照category進行排序
    clicked_articles = clicked_news.sort_values('category')

    clicked_articles['id'] = range(1, len(clicked_articles) + 1)
    # 根據獲取的id從找到candidate_news的新聞資訊
    candidate_news_id = ast.literal_eval(result.loc[user_id]['candidate_news'])
    candidate_news = news.loc[candidate_news_id]
    # 取得user的點擊預測和實際標籤
    preds = ast.literal_eval(result.loc[user_id]['predictions'])
    labels = ast.literal_eval(result.loc[user_id]['labels'])
    # 建立包含預測值和實際值的 DataFrame
    df = pd.DataFrame({'y_pred': preds, 'y_true': labels})
    # 根據預測是否正確設定顏色（綠色表示正確，紅色表示錯誤）
    df['color'] = np.where(df['y_pred'] == df['y_true'], 'green', 'red')
    # 將predict添加到candidate_news
    candidate_news['label'] = df['y_true'].values
    candidate_news['predict'] = df['y_pred'].values
    # 將顏色添加到candidate_news
    candidate_news['color'] = df['color'].values
    # 按照category進行排序
    recommend = candidate_news[candidate_news['predict'] == 1].sort_values('category')
    unrecommend = candidate_news[candidate_news['predict'] == 0].sort_values('category')
    print(unrecommend)
    return render_template('user_profile.html', user=user_id, clicked_articles=clicked_articles, recommend=recommend, unrecommend=unrecommend)

# 文字雲
@app.route('/wordcloud.png')
def wordcloud_image():
    img = io.BytesIO()

    # 從session中獲取點擊ID
    user_id =  session.get('username', None)
    
    clicked_news_id = ast.literal_eval(result.loc[user_id]['clicked_news'])
    
    clicked_news = news.loc[clicked_news_id]
    # 取得clicked_news的title
    clicked_news_titles = clicked_news['title']

    # 將所有title合併成一個字串
    wordcloud_text = ' '.join(clicked_news_titles)
    mask = np.array(Image.open("./training/dataviz/static/mask.png"))
    wordcloud = WordCloud(width=800, height=250, background_color='white', mask=mask, contour_color='white', contour_width=1).generate(wordcloud_text)
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)