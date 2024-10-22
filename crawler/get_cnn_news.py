import csv
import time
import uuid
import random
import re
import os
import pandas as pd
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

def scrape_cnn_articles():
    output_folder = 'cnn_news_output'
    os.makedirs(output_folder, exist_ok=True)
    
    driver = webdriver.Chrome()
    driver.get('https://edition.cnn.com/search?q=Artificial+Intelligence&from=0&size=10&page=1&sort=newest&types=article&section=')

    items = []
    seen = set()
    
    with open(filename, mode='a', newline='', encoding='utf-8') as new_file, \
        open('cnn_news_original.csv', mode='a', newline='', encoding='utf-8') as original_file:
        
        fieldnames = ['uuid', 'title', 'category', 'abstract', 'link', 'data_source', 'gattered_datetime','crawler_datetime','any_category']
        writer_new = csv.DictWriter(new_file, fieldnames=fieldnames)
        writer_original = csv.DictWriter(original_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer_original.writeheader()
        
        if os.stat(filename).st_size == 0:
            writer_new.writeheader()
            
        last_title, last_link = None, None
        skip_count = 0

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
                time.sleep(1)
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
                            'category': category,
                            'abstract': abstract,
                            'link': link,
                            'data_source': 'cnn_news',
                            'gattered_datetime': gattered_datetime
                        }
                        items_data.append(items_list)
                        items = [items_list]
                        
                        api_url = f"{os.environ.get('ROOT')}/api/item/crawler"
                        if api_url:  # 檢查環境變數是否存在
                            item_post = requests.post(api_url, json=items_list, timeout=20) 
                            
                            if item_post.status_code == 201:
                                recommendations = generate_random_scores(items,users)
                                time.sleep(3)
                                api_recommendations = f"{os.environ.get('ROOT')}/api/recommend/model"
                                for recommendation in recommendations:
                                    recommendations_post = requests.post(api_recommendations, json=recommendation, timeout=30) 
                                    if recommendations_post.status_code == 201:
                                        print(f"API 發送成功: {recommendations_post.text}")
            
                            if item_post.status_code != 201:
                                print(f"API 發送失敗: {item_post.text}")
                            
                        
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
    return filename 


