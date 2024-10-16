import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def init_driver():
    driver = webdriver.Chrome()
    driver.get('https://technews.tw/tag/ai')
    return driver

def wait_for_articles_to_load(driver, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, 'article'))
        )
    except TimeoutException:
        print("文章載入超時，繼續嘗試。")

def scroll_to_load(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # 等待載入完成
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def extract_article_data(tag_name_element):
    try:
        time_element = tag_name_element.find_elements(By.TAG_NAME, 'tr')[1]
        date = time_element.find_elements(By.TAG_NAME, 'span')[4].text

        title_element = tag_name_element.find_element(By.CLASS_NAME, 'entry-title')
        title = title_element.find_element(By.TAG_NAME, 'a').text
        link = title_element.find_element(By.TAG_NAME, 'a').get_attribute('href')

        abstract_element = tag_name_element.find_element(By.CLASS_NAME, "moreinf")
        abstract = abstract_element.find_elements(By.TAG_NAME, 'p')[0].text
        abstract = abstract.replace("繼續閱讀..", "").strip()

        return (date, title, link, abstract)
    except NoSuchElementException:
        return None

def scrape_articles(driver, seen):
    articles_data = []
    tag_name_elements = driver.find_elements(By.TAG_NAME, 'article')

    for tag_name_element in tag_name_elements:
        record = extract_article_data(tag_name_element)
        if record and record not in seen:
            seen.add(record)
            articles_data.append({
                'date': record[0], 'title': record[1], 
                'link': record[2], 'abstract': record[3]
            })

    return articles_data

def go_to_next_page(driver):
    try:
        next_page_button = driver.find_element(By.CLASS_NAME, 'nav-previous').find_element(By.TAG_NAME, 'a')
        next_page_button.click()
        time.sleep(3)  # 等待新頁面載入
        return True
    except NoSuchElementException:
        print("沒有更多頁面了，結束爬取。")
        return False

def save_to_csv(data, filename='output.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'title', 'link', 'abstract'])
        writer.writeheader()
        writer.writerows(data)

def main():
    driver = init_driver()
    data = []
    seen = set()

    while True:
        wait_for_articles_to_load(driver)
        scroll_to_load(driver)
        data.extend(scrape_articles(driver, seen))

        if not go_to_next_page(driver):
            break

    driver.quit()
    save_to_csv(data)

if __name__ == "__main__":
    main()

