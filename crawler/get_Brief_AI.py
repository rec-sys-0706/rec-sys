import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def initialize_driver():
    driver = webdriver.Chrome()
    return driver

def load_all_content(driver):
    """點擊 'Load More' 按鈕，直到無法再載入更多內容。"""
    while True:
        try:
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[text()="Load More"]'))
            )
            load_more_button.click()
            time.sleep(2)  
        except (NoSuchElementException, TimeoutException):
            print("無法再找到 'Load More' 按鈕")
            break

def extract_article_links(driver):
    """爬取頁面中所有文章的日期、標題和連結。"""
    class_elements = driver.find_elements(By.CLASS_NAME, 'w-full')
    data = []
    seen = set()

    for class_element in class_elements:
        try:
            time_element = class_element.find_element(By.CLASS_NAME, 'space-x-1')
            date = time_element.find_element(By.TAG_NAME, 'time').text

            title_element = class_element.find_element(By.CLASS_NAME, 'space-y-1')
            title = title_element.find_element(By.TAG_NAME, 'h2').text

            link_element = class_element.find_element(By.CLASS_NAME, 'space-y-3')
            link = link_element.find_element(By.TAG_NAME, 'a').get_attribute('href')

            record = (date, title, link)
            if record not in seen:
                seen.add(record)
                data.append({'date': date, 'title': title, 'link': link})

        except NoSuchElementException:
            continue

    return data

def extract_article_content(driver, data):
    """擷取內容。"""
    for entry in data:
        try:
            driver.get(entry['link'])
            content_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'content-blocks'))
            )
            paragraphs = content_element.find_elements(By.TAG_NAME, 'p')
            content = [p.text for p in paragraphs]
            entry['content'] = '\n'.join(content)

        except (NoSuchElementException, TimeoutException, StaleElementReferenceException):
            entry['content'] = '無法擷取內容'

def save_to_csv(data, filename='output.csv'):
    """將資料存為 CSV 檔案。"""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'title', 'link', 'content'])
        writer.writeheader()
        writer.writerows(data)

def main():
    driver = initialize_driver()
    driver.get('https://ai.briefnewsletter.com/')

    load_all_content(driver)
    articles = extract_article_links(driver)
    extract_article_content(driver, articles)

    driver.quit()
    save_to_csv(articles)
    print("爬取成功!!")

if __name__ == "__main__":
    main()
