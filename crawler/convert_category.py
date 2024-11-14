import pandas as pd

def convert_category(df):
    
    categories_to_remove = ['cnn10', 'success', 'homes', 'money']
    df = df[~df['category'].isin(categories_to_remove)]
    
    df = df.dropna(subset=['category'])

    df['category'] = df['category'].replace(['us', 'asia', 'americas', 'middleeast', 'china', 'europe', 'australia', 'world', 'africa', 'uk', 'india', 'World'], 'area-world')
    df['category'] = df['category'].replace(['health', 'health-fitness'], 'health')
    df['category'] = df['category'].replace(['politics', 'electronics'], 'electionsandpolitics')
    df['category'] = df['category'].replace(['sport', 'golf', 'football'], 'sports')
    df['category'] = df['category'].replace(['science', 'tech'], 'scienceandtechnology')
    df['category'] = df['category'].replace(['business', 'success', 'Business', 'markets'], 'business')
    df['category'] = df['category'].replace(['entertainment'], 'entertainment')
    df['category'] = df['category'].replace(['beauty', 'style', 'fashion'], 'fashion')
    df['category'] = df['category'].replace(['investing', 'economy'], 'economyandfinance')
    df['category'] = df['category'].replace(['travel', 'outdoors'], 'travel')
    df['category'] = df['category'].replace(['reviews', 'cnn-underscored', 'opinions'], 'comment')
    df['category'] = df['category'].replace(['deals', 'gifts'], 'shopping')
    df['category'] = df['category'].replace(['media', 'wbd'], 'tv')
    df['category'] = df['category'].replace(['weather', 'climate'], 'weather')
    df['category'] = df['category'].replace(['food'], 'foodanddrink')
    df['category'] = df['category'].replace(['home'], 'home')
    df['category'] = df['category'].replace(['pets'], 'petandanimal')
    df['category'] = df['category'].replace(['cars'], 'autos')

    return df