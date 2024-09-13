#文章
import pyodbc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

def setup_database_connection():
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                          'Server=SQLSERVER;'
                          'Database=SQL_DATABASE;'
                          'UID=SQL_USERNAME;'
                          'PWD=SQL_PASSWORD;'
                          'Trusted_Connection=yes;')
    return conn

def setup_webdriver(chrome_path):
    service = Service(chrome_path)
    driver = webdriver.Chrome(service=service)
    driver.implicitly_wait(10)
    return driver

def perform_search(driver, url, keyword):
    driver.get(url)
    time.sleep(2)
    driver.find_element(By.ID, 'headerSearchIcon').click()
    driver.find_element(By.CLASS_NAME, 'search-bar__input').send_keys(keyword)
    driver.find_element(By.CLASS_NAME, 'search-bar__submit').click()
    time.sleep(2)
    driver.find_element(By.XPATH, '//*[@id="search"]/div[1]/div[2]/div/div/ul/li[2]/label').click()
    time.sleep(10)

def extract_articles(driver, cur, page_count=1):
    for _ in range(page_count):
        content_div = driver.find_element(By.CLASS_NAME, 'container__field-links')
        divs_in_content = content_div.find_elements(By.XPATH, './div')
        div_count = len(divs_in_content)
        
        for x in range(1, div_count + 1):
            extract_and_save_article(driver, cur, x)
        navigate_to_next_page(driver)

def extract_and_save_article(driver, cur, index):
    titles = driver.find_element(By.XPATH, f'//*[@id="search"]/div[2]/div/div[2]/div/div[2]/div/div/div[{index}]/a[2]/div/div[1]/span').text
    publication_dates = driver.find_element(By.XPATH, f'//*[@id="search"]/div[2]/div/div[2]/div/div[2]/div/div/div[{index}]/a[2]/div/div[2]').text
    abstracts = driver.find_element(By.XPATH, f'//*[@id="search"]/div[2]/div/div[2]/div/div[2]/div/div/div[{index}]/a[2]/div/div[3]').text
    elements = driver.find_element(By.XPATH, f'//*[@id="search"]/div[2]/div/div[2]/div/div[2]/div/div/div[{index}]/a[2]')
    urls = elements.get_attribute('href')

    driver.get(urls)
    categories, subcategories = extract_categories(driver)
    content = extract_content(driver)

    cur.execute('''INSERT INTO stories(category,subcategory,title,date,abstract,contents,url) 
                   VALUES(?,?,?,?,?,?,?);''', (categories, subcategories, titles, publication_dates, abstracts, content, urls))
    cur.connection.commit()
    driver.back()
    time.sleep(1)

def extract_categories(driver):
    try:
        categories = driver.find_element(By.CLASS_NAME, 'breadcrumb__parent-link').text
        subcategories = driver.find_element(By.CLASS_NAME, 'breadcrumb__child-link').text
    except:
        categories = " "
        subcategories = " "
    return categories, subcategories

def extract_content(driver):
    try:
        article_content = driver.find_element(By.CLASS_NAME, 'article__content')
        time.sleep(2)
        paragraphs = article_content.find_elements(By.TAG_NAME, 'p')
        time.sleep(2)
        content = "\n".join([p.text for p in paragraphs])
    except:
        content = " "
    return content

def navigate_to_next_page(driver):
    try:
        driver.find_element(By.XPATH, '//*[@id="search"]/div[2]/div/div[4]/div/div[3]').click()
        time.sleep(10)
    except:
        print("No more pages.")

def main():
    conn = setup_database_connection()
    cur = conn.cursor()
    cur.fast_executemany = True

    chrome_path = "../chromedriver-win32/chromedriver.exe"

    driver = setup_webdriver(chrome_path)

    url = 'https://edition.cnn.com/'
    keyword = 'Artificial Intelligence'
    
    perform_search(driver, url, keyword)
    extract_articles(driver, cur, page_count=4)  # 控制頁數
    driver.quit()

if __name__ == '__main__':
    main()