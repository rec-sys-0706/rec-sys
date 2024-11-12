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
from urllib.parse import urlparse

def scrape_huggingface_papers():
    driver = webdriver.Chrome()
    driver.get('https://huggingface.co/papers')
    
    file_exists = os.path.isfile('daily_papers.csv')
    
    if file_exists:
        existing_data = pd.read_csv('daily_papers.csv')
        existing_titles_links = set(zip(existing_data['title'], existing_data['link']))
    else:
        existing_data = pd.DataFrame(columns=['title', 'link'])
        existing_titles_links = set()
    
    with open('daily_papers.csv', mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ['uuid', 'title', 'category', 'abstract', 'link', 'data_source', 'gattered_datetime','crawler_datetime','any_category']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
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
                    crawler_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(crawler_datetime)
                        
                    unique_id = str(uuid.uuid4())
                    print(unique_id)
                        
                    try:
                        title_element = article_element.find_element(By.TAG_NAME, 'h3')
                        title = title_element.text
                        print(title)
                            
                        link = title_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                        print(link)
                    except StaleElementReferenceException:
                        print("元素已失效，重新嘗試抓取。")
                        continue    
                        
                    if existing_data[(existing_data['title'] == title) & (existing_data['link'] == link)].empty:    
                        arxiv = 'https://arxiv.org/abs/' + re.sub('/papers/', '', urlparse(link).path)
                        driver.execute_script(f"window.open('{arxiv}', '_blank');")
                        driver.switch_to.window(driver.window_handles[1])
                            
                        try:
                            abstract = WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.CLASS_NAME, 'abstract'))
                            ).text
                            abstract = abstract.replace("\n", " ")
                            print(abstract)
                                
                            gattered_datetime_element = driver.find_element(By.ID, 'content-inner')
                            gattered_datetime_original = gattered_datetime_element.find_element(By.CLASS_NAME, 'dateline').text
                                
                            pattern = r'\d{1,2} \w{3} \d{4}'
                            match = re.search(pattern, gattered_datetime_original)
                            date_str = match.group(0) if match else None
                            date_obj = datetime.strptime(date_str, "%d %b %Y")
                            gattered_datetime = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                            print(gattered_datetime)
                            
                        except (NoSuchElementException, TimeoutException) as e:
                            print(f"跳過該元素，原因：{e}")
                                
                        category = ""
                        any_category = "NO"
                        print(category)
                        print(any_category)
                                
                        data_source = 'hf_paper'
                        print(data_source)
                        
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        time.sleep(1)
                
                        writer.writerow({
                            'uuid': unique_id,
                            'title': title,
                            'category': category,
                            'abstract': abstract,
                            'link': link,
                            'data_source': data_source,
                            'gattered_datetime': gattered_datetime,
                            'crawler_datetime': crawler_datetime,
                            'any_category': any_category
                        })
                                
                        file.flush()
                        existing_titles_links.add((title, link))
                        
                    else:
                        print(f"資料已存在，跳過標題: {title}")
                            
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

scrape_huggingface_papers()