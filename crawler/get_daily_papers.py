import csv
import time
import uuid
import random
import pandas as pd
import json
import re
import os
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, 
    StaleElementReferenceException)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
from recommendation import generate_random_scores
import os, requests

def scrape_huggingface_papers(output_file='output5.csv'):
    # response = requests.get(f"{os.environ.get('ROOT')}/api/user")
    # data = response.json()
    # users = data["data"]
    
    driver = webdriver.Chrome()
    driver.get('https://huggingface.co/papers')
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_folder, f'daily_papers_{current_time}.csv')
    
    file_exists = os.path.isfile('daily_papers_original.csv')
    
    if file_exists:
        existing_data = pd.read_csv('daily_papers_original.csv')
        existing_titles_links = set(zip(existing_data['title'], existing_data['link']))
    else:
        existing_data = pd.DataFrame(columns=['title', 'link'])
        existing_titles_links = set()
    
    with open(filename, mode='a', newline='', encoding='utf-8') as new_file, \
        open('daily_papers_original.csv', mode='a', newline='', encoding='utf-8') as original_file:
        
        fieldnames = ['uuid', 'title', 'category', 'abstract', 'link', 'data_source', 'gattered_datetime','crawler_datetime','any_category']
        writer_new = csv.DictWriter(new_file, fieldnames=fieldnames)
        writer_original = csv.DictWriter(original_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer_original.writeheader()
        
        if os.stat(filename).st_size == 0:
            writer_new.writeheader()
        
        while True:
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    print("已達頁面底部，無更多內容。")
                    break
                last_height = new_height
                
            try:
                articles_elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.TAG_NAME, 'article'))
                )
            except TimeoutException:
                print("文章載入超時，結束爬取。")
                driver.quit()
                return

        for article_element in articles_elements:
            try:
                # 捕捉並處理 StaleElementReferenceException
                try:
                    title_element = article_element.find_element(By.TAG_NAME, 'h3')
                    title = title_element.text
                    print(title)
                    link = title_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    print(link)
                except StaleElementReferenceException:
                    print("元素已失效，重新嘗試抓取。")
                    continue

                # 進入 arXiv 網頁抓取摘要與日期
                arxiv = 'https://arxiv.org/abs/' + re.sub('/papers/', '', urlparse(link).path)
                driver.execute_script(f"window.open('{arxiv}', '_blank');")
                driver.switch_to.window(driver.window_handles[1])

                try:
                    abstract = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'abstract'))
                    ).text
                    print(abstract)
                    gattered_datetime_element = driver.find_element(By.ID, 'content-inner')
                    gattered_datetime_original = gattered_datetime_element.find_element(By.CLASS_NAME, 'dateline').text

                    # 使用正則表達式提取日期
                    pattern = r'\d{1,2} \w{3} \d{4}'
                    match = re.search(pattern, gattered_datetime_original)
                    date_str = match.group(0) if match else None
                    date_obj = datetime.strptime(date_str, "%d %b %Y")
                    gattered_datetime = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    print(gattered_datetime)

                    # 防止重複記錄
                    record = (title, abstract, link, gattered_datetime)
                    
                    if title and abstract and link and gattered_datetime:
                        if record not in seen:
                            seen.add(record)
                            items_list = {
                                'uuid': str(uuid.uuid4()),
                                'title': title,
                                'abstract': abstract,
                                'link': link,
                                'data_source': 'hf_paper',
                                'gattered_datetime': gattered_datetime
                            }
                            items_data.append(items_list)    
                            items = [items_list]
                                                
                            # api_url = f"{os.environ.get('ROOT')}/api/item/crawler"
                            # if api_url:  # 檢查環境變數是否存在
                            #     item_post = requests.post(api_url, json=items_list, timeout=20) 
                                
                            #     if item_post.status_code == 201:
                            #         recommendations = generate_random_scores(items,users)
                            #         time.sleep(3)
                            #         api_recommendations = f"{os.environ.get('ROOT')}/api/recommend/model"
                            #         for recommendation in recommendations:
                            #             recommendations_post = requests.post(api_recommendations, json=recommendation, timeout=30) 
                            #             if recommendations_post.status_code == 201:
                            #                 print(f"API 發送成功: {recommendations_post.text}")
                
                            #     if item_post.status_code != 201:
                            #         print(f"API 發送失敗: {item_post.text}")
                                
                            
                            with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                                writer = csv.DictWriter(file, fieldnames=fieldnames)
                                writer.writerow(items_list)
                    else:
                        print(f"缺少資料: title={title}, abstract={abstract}, link={link}, gattered_datetime={gattered_datetime}")
                
                except (NoSuchElementException, TimeoutException) as e:
                    print(f"跳過該元素，原因：{e}")
                    
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(1)    
                
                        row_data = {
                            'uuid': unique_id,
                            'title': title,
                            'category': category,
                            'abstract': abstract,
                            'link': link,
                            'data_source': data_source,
                            'gattered_datetime': gattered_datetime,
                            'crawler_datetime': crawler_datetime,
                            'any_category': any_category
                        }
                             
                        writer_new.writerow(row_data)
                        writer_original.writerow(row_data) 
                          
                        new_file.flush()
                        original_file.flush()
                        
                        existing_titles_links.add((title, link))
                        
                    else:
                        print(f"資料已存在，跳過標題: {title}")
                        driver.quit()
                        return filename
                            
                except (NoSuchElementException, StaleElementReferenceException) as e:
                    print(f"遇到錯誤，跳過該項：{e}")
                    continue
                    
            try:
                next_page_button = driver.find_element(By.LINK_TEXT, 'Previous')
                next_page_button.click()
                time.sleep(1)
            except NoSuchElementException:
                print("沒有更多頁面了，結束爬取。")
                break
            
    driver.quit()
    return filename