import pandas as pd

def convert_category(df):
    
    categories_to_remove = ['cnn10', 'success', 'homes', 'money']
    df = df[~df['category'].isin(categories_to_remove)]

    df['category'] = df['category'].replace(['us', 'asia', 'americas', 'middleeast', 'china', 'europe', 'australia', 'world', 'africa', 'uk', 'india', 'World'], 'other_area')
    df['category'] = df['category'].replace(['health', 'health-fitness'], 'health')
    df['category'] = df['category'].replace(['politics', 'electronics'], 'politics')
    df['category'] = df['category'].replace(['sport', 'golf', 'football'], 'sports')
    df['category'] = df['category'].replace(['science', 'tech'], 'science-and-technology')
    df['category'] = df['category'].replace(['business', 'success', 'Business', 'markets', 'energy'], 'business')
    df['category'] = df['category'].replace(['entertainment'], 'entertainment')
    df['category'] = df['category'].replace(['beauty', 'style', 'fashion'], 'fashion')
    df['category'] = df['category'].replace(['investing', 'economy'], 'economy-and-finance')
    df['category'] = df['category'].replace(['travel', 'outdoors'], 'travel')
    df['category'] = df['category'].replace(['reviews', 'cnn-underscored', 'opinions'], 'comment')
    df['category'] = df['category'].replace(['deals', 'gifts'], 'shopping')
    df['category'] = df['category'].replace(['media', 'wbd'], 'tv')
    df['category'] = df['category'].replace(['weather', 'climate'], 'weather')
    df['category'] = df['category'].replace(['food'], 'food-and-drink')
    df['category'] = df['category'].replace(['home'], 'home')
    df['category'] = df['category'].replace(['pets'], 'pet-and-animal')
    df['category'] = df['category'].replace(['cars'], 'autos')

    return df

# 讀取資料
df = pd.read_csv('cnn_news_1.csv')
# 轉換分類
df = convert_category(df)
# 檢查轉換後的分類
print(df['category'].unique())
# 儲存結果
df.to_csv('cnn_news_1.csv', index=False)