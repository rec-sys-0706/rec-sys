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
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_technews_ai():
    driver = webdriver.Chrome()
    driver.get('https://technews.tw/tag/ai')

    items = []
    seen = set()

    while True:  
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, 'article'))
            )
        except TimeoutException:
            print("文章載入超時，繼續嘗試。")

        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)  
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        tag_name_elements = driver.find_elements(By.TAG_NAME, 'article')

        for tag_name_element in tag_name_elements:
            try:
                title_element = tag_name_element.find_element(By.CLASS_NAME, 'entry-title')
                title = title_element.find_element(By.TAG_NAME, 'a').text
                
                abstract_element = tag_name_element.find_element(By.CLASS_NAME, "moreinf")
                abstract = abstract_element.find_elements(By.TAG_NAME, 'p')[0].text
                abstract = abstract.replace("繼續閱讀..", "").strip()
                
                link = title_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                
                gattered_datetime_element = tag_name_element.find_elements(By.TAG_NAME, 'tr')[1]
                date_str = gattered_datetime_element.find_elements(By.TAG_NAME, 'span')[4].text
                date_obj = datetime.strptime(date_str, "%Y 年 %m 月 %d 日 %H:%M")
                gattered_datetime = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                
                record = (title, abstract, link, gattered_datetime)
                if record not in seen:  
                    seen.add(record)
                    item_data = {
                        'title': title,
                        'abstract': abstract,
                        'link': link,
                        'data_source': 'tech_news',  
                        'gattered_datetime': gattered_datetime
                    }
                    items.append(item_data)
                    
                    api_url = f"{os.environ.get('ROOT')}:5000/api/item/crawler"
                    if api_url:  # 檢查環境變數是否存在
                        item_post = requests.post(api_url, json=item_data, timeout=30) 
                        if item_post.status_code != 201:
                            print(f"API 發送失敗: {item_post.text}")
                        if item_post.status_code == 201:
                            print(f"API 發送成功: {item_post.text}")

            except NoSuchElementException:
                continue

        try:
            next_page_button = driver.find_element(By.CLASS_NAME, 'nav-previous').find_element(By.TAG_NAME, 'a')
            next_page_button.click()
            time.sleep(3)  
        except NoSuchElementException:
            print("沒有更多頁面了，結束爬取。")
            break  

    driver.quit()

scrape_technews_ai()
