import csv
import time
import uuid
import random
import requests
import pandas as pd
import json
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, 
    StaleElementReferenceException, ElementClickInterceptedException
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_cnn_articles(output_file='output7.csv'):
    driver = webdriver.Chrome()
    driver.get('https://edition.cnn.com/search?q=Artificial+Intelligence&from=0&size=10&page=1&sort=newest&types=article&section=')

    items = []
    seen = set()
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['uuid', 'title', 'abstract', 'link', 'data_source', 'gattered_datetime']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    while True:
        try:
            class_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'container__item'))
            )
        except TimeoutException:
            print("無法載入頁面內容，結束程式。")
            break

        for class_element in class_elements:
            try:
                text_container = class_element.find_element(By.CLASS_NAME, 'container__text')
                date_str = text_container.find_element(By.CLASS_NAME, 'container__date').text
                date_obj = datetime.strptime(date_str, "%b %d, %Y")
                gattered_datetime = date_obj.strftime("%Y-%m-%d 00:00:00")
                
                title = text_container.find_element(By.CLASS_NAME, 'container__headline-text').text

                abstract_element = class_element.find_element(By.CLASS_NAME, 'container__description')
                abstract = abstract_element.text

                link = class_element.find_element(By.CLASS_NAME, 'container__link').get_attribute('href')

                record = (title, abstract, link, gattered_datetime)
                
                if title and abstract and link and gattered_datetime:
                    if record not in seen:
                        seen.add(record)
                        item_data = {
                            'uuid': str(uuid.uuid4()),
                            'title': title,
                            'abstract': abstract,
                            'link': link,
                            'data_source': 'bbc_news',
                            'gattered_datetime': gattered_datetime
                        }
                        items.append(item_data)
                        
                        # api_url = f"{os.environ.get('ROOT')}:5000/api/item/crawler"
                        # if api_url:  # 檢查環境變數是否存在
                        #     item_post = requests.post(api_url, json=item_data, timeout=10) 
                        #     if item_post.status_code != 201:
                        #         print(f"API 發送失敗: {item_post.text}")
                        #     if item_post.status_code == 201:
                        #         print(f"API 發送成功: {item_post.text}")
                        
                        with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                                writer = csv.DictWriter(file, fieldnames=fieldnames)
                                writer.writerow(item_data)
                                
                else:
                        print(f"缺少資料: title={title}, abstract={abstract}, link={link}, gattered_datetime={gattered_datetime}")
                
            except (NoSuchElementException, StaleElementReferenceException) as e:
                print(f"遇到錯誤，跳過該項：{e}")
                continue

        try:
            next_page_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="search"]/div[2]/div/div[4]/div/div[3]'))
            )
            next_page_button.click()
            time.sleep(random.uniform(2, 5))  # 隨機延遲控制速度
        except (NoSuchElementException, TimeoutException):
            print("沒有找到下一頁，停止爬取。")
            break
        except ElementClickInterceptedException:
            print("無法點擊下一頁按鈕，可能是其他元素覆蓋了它。")
            break

    driver.quit()

