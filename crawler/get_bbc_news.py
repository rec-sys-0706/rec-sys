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

def scrape_bbc_articles(output_file='output6.csv'):
    driver = webdriver.Chrome()
    driver.get('https://www.bbc.com/search?q=Artificial%20Intelligence&edgeauth=eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJrZXkiOiAiZmFzdGx5LXVyaS10b2tlbi0xIiwiZXhwIjogMTcyOTQzMDIyMiwibmJmIjogMTcyOTQyOTg2MiwicmVxdWVzdHVyaSI6ICIlMkZzZWFyY2glM0ZxJTNEQXJ0aWZpY2lhbCUyNTIwSW50ZWxsaWdlbmNlIn0.mUpWVN3N9gvpw2DpJz8UhFLrxTbWLb17ZWLp81mNGi4')

    items = []
    seen = set()
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['uuid', 'title', 'abstract', 'link', 'data_source', 'gattered_datetime']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    while True:
        try:
            original_element = driver.find_element(By.CLASS_NAME, 'sc-32f23d22-1')
            class_element = original_element.find_element(By.CLASS_NAME, 'sc-32f23d22-7')
            div_elements = class_element.find_elements(By.XPATH, './div')
            
        except TimeoutException:
            print("無法載入頁面內容，結束程式。")
            break

        for div_element in div_elements:
            try:
                title = div_element.find_element(By.CLASS_NAME, 'sc-1207bea1-3').text
                print(title)

                abstract = div_element.find_element(By.CLASS_NAME, 'sc-ae29827d-0').text
                print(abstract)
                
                link = div_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                print(link)

                try:
                    date_str = div_element.find_element(By.CLASS_NAME, 'sc-4e537b1-1').text
                    date_obj = datetime.strptime(date_str, "%d %b %Y")
                    gattered_datetime = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    print(gattered_datetime)
                except ValueError:
                    print(f"日期格式錯誤，跳過：{date_str}")
                    continue  # 跳過該筆資料
                
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
            next_page_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Next")]'))
            )
            next_page_button.click()

        except (NoSuchElementException, TimeoutException):
            print("沒有找到下一頁，停止爬取。")
            break

    driver.quit()