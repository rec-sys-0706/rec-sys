import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def init_driver(url):
    """初始化WebDriver並打開指定的URL"""
    driver = webdriver.Chrome()
    driver.get(url)
    return driver

def load_more(driver):
    """點擊 'Load More' 按鈕，直到無法再找到該按鈕"""
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

def scrape_data(driver):
    """從網頁中提取數據"""
    data = []
    seen = set()

    class_elements = driver.find_elements(By.CLASS_NAME, 'w-full')
    
    for class_element in class_elements:
        try:
            time_element = class_element.find_element(By.CLASS_NAME, 'space-x-1')
            date = time_element.find_element(By.TAG_NAME, 'time').text
            
            title_element = class_element.find_element(By.CLASS_NAME, 'space-y-1')
            title = title_element.find_element(By.TAG_NAME, 'h2').text
             
            link_element = class_element.find_element(By.CLASS_NAME, 'space-y-3')
            link_element1 = link_element.find_element(By.TAG_NAME, 'a')
            link = link_element1.get_attribute('href')
            
            record = (date, title, link)
            if record not in seen:  
                seen.add(record) 
                data.append({'date': date, 'title': title, 'link': link})

        except NoSuchElementException:
            continue

    return data

def save_to_csv(data, filename):
    """將數據保存到CSV文件中"""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'title', 'link'])
        writer.writeheader()
        writer.writerows(data)

def main():
    url = 'https://ai.briefnewsletter.com/'
    driver = init_driver(url)
    
    load_more(driver)
    
    data = scrape_data(driver)
    
    driver.quit()
    
    save_to_csv(data, 'output.csv')

if __name__ == "__main__":
    main()
