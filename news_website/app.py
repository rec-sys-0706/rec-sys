from flask import Flask, render_template, send_file
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Sample news data
articles = [
    {
        'category': 'Breaking News',
        'title': 'Breaking News: Market Hits All-Time High',
        'content': 'The stock market has reached an all-time high today, with major indices showing significant gains.',
        'author': 'John Doe',
        'date': '2024-08-09'
    },
    {
        'category': 'Tech Innovations',
        'title': 'Tech Innovations: AI Revolutionizing Industries',
        'content': 'Artificial Intelligence is transforming the way businesses operate, from automation to customer service.',
        'author': 'Jane Smith',
        'date': '2024-08-08'
    },
    {
        'category': 'Breaking News',
        'title': 'Breaking News: Market Hits All-Time High',
        'content': 'The stock market has reached an all-time high today, with major indices showing significant gains.',
        'author': 'John Doe',
        'date': '2024-08-09'
    },
    {
        'category': 'Tech Innovations',
        'title': 'Tech Innovations: AI Revolutionizing Industries',
        'content': 'Artificial Intelligence is transforming the way businesses operate, from automation to customer service.',
        'author': 'Jane Smith',
        'date': '2024-08-08'
    },
    {
        'category': 'Health Update',
        'title': 'Health Update: New Breakthrough in Cancer Research',
        'content': 'Scientists have announced a major breakthrough in cancer treatment, promising better outcomes for patients.',
        'author': 'Alice Brown',
        'date': '2024-08-07'
    },
    {
        'category': 'Global Politics',
        'title': 'Global Politics: Tensions Rise Between Major Powers',
        'content': 'International tensions have escalated following new sanctions imposed by leading nations.',
        'author': 'Robert White',
        'date': '2024-08-06'
    },
    {
        'category': 'Environment',
        'title': 'Environment: Climate Change Impacts Coastal Cities',
        'content': 'Rising sea levels are beginning to affect coastal communities, with increased flooding and erosion.',
        'author': 'Emily Green',
        'date': '2024-08-05'
    },
    {
        'category': 'Sports',
        'title': 'Sports: Historic Win at the World Championship',
        'content': 'An underdog team has won the world championship in a thrilling final match, shocking fans worldwide.',
        'author': 'Michael Black',
        'date': '2024-08-04'
    },
    {
        'category': 'Entertainment',
        'title': 'Entertainment: Blockbuster Movie Breaks Box Office Records',
        'content': 'The latest summer blockbuster has shattered previous box office records, grossing over $1 billion in its first week.',
        'author': 'Sarah Blue',
        'date': '2024-08-03'
    },
    {
        'category': 'Economy',
        'title': 'Economy: Experts Predict Economic Recession',
        'content': 'Economists are warning of a potential recession as economic indicators show signs of a slowdown.',
        'author': 'David Grey',
        'date': '2024-08-02'
    },
    {
        'category': 'Technology',
        'title': 'Technology: New Smartphone Model Unveiled',
        'content': 'A leading tech company has just unveiled its latest smartphone model, boasting innovative features and improved performance.',
        'author': 'Laura Silver',
        'date': '2024-08-01'
    },
    {
        'category': 'Travel',
        'title': 'Travel: Top Destinations for 2024',
        'content': 'Travel experts have released their list of top travel destinations for 2024, highlighting must-visit locations around the globe.',
        'author': 'Tom Gold',
        'date': '2024-07-31'
    },
    {
        'category': 'Science',
        'title': 'Science: Discovery of a New Exoplanet',
        'content': 'Astronomers have discovered a new exoplanet that may have the conditions necessary to support life.',
        'author': 'Sophia White',
        'date': '2024-07-30'
    },
    {
        'category': 'Fashion',
        'title': 'Fashion: Latest Trends for the Fall Season',
        'content': 'The fashion industry is buzzing with the latest trends for the fall season, featuring bold colors and innovative designs.',
        'author': 'Isabella Green',
        'date': '2024-07-29'
    },
    {
        'category': 'Education',
        'title': 'Education: New Policies in Public Schools',
        'content': 'Public schools are set to implement new policies aimed at improving student outcomes and reducing inequality.',
        'author': 'Liam Brown',
        'date': '2024-07-28'
    },
    {
        'category': 'Automotive',
        'title': 'Automotive: Electric Vehicles Gaining Popularity',
        'content': 'The market for electric vehicles is growing rapidly as more consumers embrace eco-friendly transportation options.',
        'author': 'Oliver Grey',
        'date': '2024-07-27'
    },
    {
        'category': 'Food',
        'title': 'Food: The Rise of Plant-Based Diets',
        'content': 'Plant-based diets are becoming increasingly popular as people seek healthier and more sustainable eating habits.',
        'author': 'Emma Silver',
        'date': '2024-07-26'
    }
    # Add more news articles here
]

categories = [article['category'] for article in articles]
wordcloud_text = ' '.join(categories)

mask = np.array(Image.open("static/mask.png"))
wordcloud = WordCloud().generate(wordcloud_text)
# Save the wordcloud to a BytesIO object
img = io.BytesIO()
wordcloud.to_image().save(img, format='PNG')
img.seek(0)


@app.route('/')
def index():
    return render_template('index.html', img = img, articles=articles)

@app.route('/wordcloud.png')
def wordcloud_image():
    img = io.BytesIO()
    categories = [article['category'] for article in articles]
    wordcloud_text = ' '.join(categories)
    mask = np.array(Image.open("static/mask.png"))
    wordcloud = WordCloud(width=800, height=400, background_color='transparent', mask=mask, contour_color='white', contour_width=1).generate(wordcloud_text)
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
