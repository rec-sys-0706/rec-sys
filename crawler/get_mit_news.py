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
from recommendation import generate_random_scores

def scrape_mit_news_articles(output_file='output8.csv'):
    driver = webdriver.Chrome()
    driver.get('https://news.mit.edu/topic/artificial-intelligence2')

    items_data = []
    seen = set()
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['uuid', 'title', 'abstract', 'link', 'data_source', 'gattered_datetime']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


    while True:
        try:
            class_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'page-term--views--list-item'))
            )
        except TimeoutException:
            print("無法載入頁面內容，結束程式。")
            break

        for class_element in class_elements:
            try:
                
                title = class_element.find_element(By.CLASS_NAME, 'term-page--news-article--item--title--link').text
                print(title)

                abstract = class_element.find_element(By.CLASS_NAME, 'term-page--news-article--item--dek').text
                print(abstract)

                link = class_element.find_element(By.CLASS_NAME, 'term-page--news-article--item--title--link').get_attribute('href')
                print(link)

                date_str = class_element.find_element(By.CLASS_NAME, 'term-page--news-article--item--publication-date').text
                date_obj = datetime.strptime(date_str, "%B %d, %Y")
                gattered_datetime = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                print(gattered_datetime)
                
                record = (title, abstract, link, gattered_datetime)
                
                if title and abstract and link and gattered_datetime:
                    if record not in seen:
                        seen.add(record)
                        items = {
                            'uuid': str(uuid.uuid4()),
                            'title': title,
                            'abstract': abstract,
                            'link': link,
                            'data_source': 'mit_news',
                            'gattered_datetime': gattered_datetime
                        }
                        items_data.append(items)
                        
                        api_url = f"{os.environ.get('ROOT')}:5000/api/item/crawler"
                        if api_url:  
                            item_post = requests.post(api_url, json=items, timeout=10) 
                            if item_post.status_code == 201:
                                generate_random_scores(items,users)
                            if item_post.status_code != 201:
                                print(f"API 發送失敗: {item_post.text}")
                            
                        
                        with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.DictWriter(file, fieldnames=fieldnames)
                            writer.writerow(items)
                        
                else:
                        print(f"缺少資料: title={title}, abstract={abstract}, link={link}, gattered_datetime={gattered_datetime}")
                
            except (NoSuchElementException, StaleElementReferenceException) as e:
                print(f"遇到錯誤，跳過該項：{e}")
                continue

        try:
            next_page_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'pager--button--next'))
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