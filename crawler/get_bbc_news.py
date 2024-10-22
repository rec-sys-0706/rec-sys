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

def scrape_bbc_articles(output_file='output6.csv'):
    response = requests.get(f"{os.environ.get('ROOT')}/api/user")
    data = response.json()
    users = data["data"]
    
    driver = webdriver.Chrome()
    driver.get('https://www.bbc.com/search?q=Artificial%20Intelligence&edgeauth=eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJrZXkiOiAiZmFzdGx5LXVyaS10b2tlbi0xIiwiZXhwIjogMTcyOTQzMDIyMiwibmJmIjogMTcyOTQyOTg2MiwicmVxdWVzdHVyaSI6ICIlMkZzZWFyY2glM0ZxJTNEQXJ0aWZpY2lhbCUyNTIwSW50ZWxsaWdlbmNlIn0.mUpWVN3N9gvpw2DpJz8UhFLrxTbWLb17ZWLp81mNGi4')

    items_data = []
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
                time.sleep(1)
                title = div_element.find_element(By.CLASS_NAME, 'sc-8ea7699c-3').text
                #print(title)

                abstract = div_element.find_element(By.CLASS_NAME, 'sc-ae29827d-0').text
                #print(abstract)
                
                link = div_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                #print(link)

                try:
                    date_str = div_element.find_element(By.CLASS_NAME, 'sc-6fba5bd4-1').text
                    date_obj = datetime.strptime(date_str, "%d %b %Y")
                    gattered_datetime = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    #print(gattered_datetime)
                except ValueError:
                    print(f"日期格式錯誤，跳過：{date_str}")
                    continue  # 跳過該筆資料
                
                record = (title, abstract, link, gattered_datetime)
                
                if title and abstract and link and gattered_datetime:
                    if record not in seen:
                        seen.add(record)
                        items_list = {
                            'uuid': str(uuid.uuid4()),
                            'title': title,
                            'abstract': abstract,
                            'link': link,
                            'data_source': 'bbc_news',
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
                            writer.writerow(items_list)
                        
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
    
scrape_bbc_articles()
    