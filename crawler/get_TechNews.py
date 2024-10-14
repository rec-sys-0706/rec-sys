import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def initialize_driver():
    driver = webdriver.Chrome()
    return driver

def load_page(driver, url):
    driver.get(url)

def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def extract_articles(driver, seen):
    data = []
    articles = driver.find_elements(By.TAG_NAME, 'article')

    for article in articles:
        try:
            date = article.find_elements(By.TAG_NAME, 'tr')[1] \
                          .find_elements(By.TAG_NAME, 'span')[4].text
            title = article.find_element(By.CLASS_NAME, 'entry-title') \
                           .find_element(By.TAG_NAME, 'a').text
            link = article.find_element(By.CLASS_NAME, 'entry-title') \
                          .find_element(By.TAG_NAME, 'a').get_attribute('href')
            abstract = article.find_element(By.CLASS_NAME, 'moreinf') \
                              .find_elements(By.TAG_NAME, 'p')[0].text
            abstract = abstract.replace("繼續閱讀..", "").strip()

            record = (date, title, link, abstract)
            if record not in seen:
                seen.add(record)
                data.append({'date': date, 'title': title, 'link': link, 'abstract': abstract})
        
        except NoSuchElementException:
            continue

    return data

def click_next_page(driver):
    try:
        driver.find_element(By.CLASS_NAME, 'nextpage').click()
        return True
    except NoSuchElementException:
        return False

def save_to_csv(data, filename='output.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'title', 'link', 'abstract'])
        writer.writeheader()
        writer.writerows(data)

def main():
    driver = initialize_driver()
    seen = set()
    all_data = []

    try:
        load_page(driver, 'https://cdn.technews.tw/')
        
        for _ in range(5):
            scroll_to_bottom(driver)
            articles = extract_articles(driver, seen)
            all_data.extend(articles)

            if not click_next_page(driver):
                break  # 若無下一頁，結束迴圈

    finally:
        driver.quit()

    save_to_csv(all_data)

if __name__ == "__main__":
    main()
