import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pyodbc
from datetime import datetime

folder_path = "C:/Users/user/Desktop/website/news"
file_path = os.path.join(folder_path, "CNN_news.txt")
backup_file_path = os.path.join(folder_path, "news_database_backup.txt")

os.makedirs(folder_path, exist_ok=True)

url = "https://edition.cnn.com/"

categories = {
    'us': 'US',
    'world': 'World',
    'politics': 'Politics',
    'business': 'Business',
    'opinion': 'Opinion',
    'health': 'Health',
    'entertainment': 'Entertainment',
    'style': 'Style',
    'travel': 'Travel',
    'sports': 'Sports',
    'science': 'Science',
    'climate': 'Climate',
    'weather': 'Weather',
    'ukraine-russia-war': 'Ukraine-Russia War',
    'israel-hamas-war': 'Israel-Hamas War',
    'paris-olympics': 'Paris Olympics'
}

try:
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.content
    print("Successfully fetched homepage content")
except requests.RequestException as e:
    print(f"Error fetching homepage: {str(e)}")
    html_content = None

def extract_category_and_subcategory(full_link, news_html):
    for key in categories:
        if key in full_link:
            return categories[key], key
    section = news_html.find('meta', {'name': 'section'})
    subsection = news_html.find('meta', {'name': 'subsection'})
    if section and subsection:
        return section.get('content', 'General'), subsection.get('content', 'General')
    elif section:
        return section.get('content', 'General'), 'General'
    return "General", "General"

if html_content:
    soup = BeautifulSoup(html_content, "html.parser")

    found_articles = 0
    links = soup.find_all("a", href=True)

    with open(file_path, "w", encoding="utf-8") as file, open(backup_file_path, "w", encoding="utf-8") as backup_file:
        for link in links:
            if found_articles >= 30:
                break
            href = link.get("href")
            full_link = urljoin(url, href)

            if "/2024/" in full_link and "/index.html" in full_link:
                print(f"Processing link: {full_link}")
                try:
                    response = requests.get(full_link)
                    response.raise_for_status()
                    news_html = BeautifulSoup(response.content, "html.parser")

                    title = news_html.find('h1').get_text() if news_html.find('h1') else 'No Title'
                    paragraphs = news_html.find_all('p')
                    abstract = ' '.join([p.get_text() for p in paragraphs[:2]])
                    content = ' '.join([p.get_text() for p in paragraphs])

                    category, subcategory = extract_category_and_subcategory(full_link, news_html)
                    publication_date = news_html.find('meta', {'name': 'pubdate'}).get('content') if news_html.find('meta', {'name': 'pubdate'}) else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # SQL Server connection 
                    conn_str = (
                        "Driver={SQL Server};"
                        "Server=LAPTOP-IGBO7T9O\\SQLEXPRESS01;"
                        "Database=cnn;"
                    )

                    conn = pyodbc.connect(conn_str)

                    # Check if the URL already exists
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM News WHERE URL = ?", full_link)
                    count = cursor.fetchone()[0]
                    cursor.close()

                    if count == 0:
                        # Write news content to file
                        file.write(f"Category: {category}\n")
                        file.write(f"Subcategory: {subcategory}\n")
                        file.write(f"Title: {title}\n\n")
                        file.write(f"Abstract: {abstract}\n\n")
                        file.write(f"Content: {content}\n\n")
                        file.write(f"URL: {full_link}\n\n")
                        file.write(f"Publication Date: {publication_date}\n\n")

                        # Insert data into SQL Server
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO News (Category, Subcategory, Title, Abstract, Content, URL, PublicationDate)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, category, subcategory, title, abstract, content, full_link, publication_date)
                        conn.commit()
                        cursor.close()

                        backup_file.write(f"Category: {category}\n")
                        backup_file.write(f"Subcategory: {subcategory}\n")
                        backup_file.write(f"Title: {title}\n\n")
                        backup_file.write(f"Abstract: {abstract}\n\n")
                        backup_file.write(f"Content: {content}\n\n")
                        backup_file.write(f"URL: {full_link}\n\n")
                        backup_file.write(f"Publication Date: {publication_date}\n\n")

                        print(f"News inserted into SQL Server and backed up! ({found_articles + 1}/30)")
                        found_articles += 1
                    else:
                        print(f"News with URL {full_link} already exists, skipping insertion.")

                    conn.close()

                except requests.RequestException as e:
                    print(f"Error requesting {full_link}: {str(e)}")

    if found_articles == 0:
        print("No valid news links found")

else:
    print("Homepage content is empty, cannot proceed")
