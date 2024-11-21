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

<<<<<<< HEAD
def scrape_cnn_articles(output_file='output7.csv'):
    response = requests.get(f"{os.environ.get('ROOT')}/api/user")
    data = response.json()
    users = data["data"]
    
    driver = webdriver.Chrome()
    driver.get('https://edition.cnn.com/search?q=Artificial+Intelligence&from=0&size=10&page=1&sort=newest&types=article&section=')

    items_data = []
    seen = set()
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['uuid', 'title', 'abstract', 'link', 'data_source', 'gattered_datetime']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
=======
def scrape_cnn_articles():
    output_folder = 'cnn_news_output'
    os.makedirs(output_folder, exist_ok=True)
    
    driver = webdriver.Chrome()
    driver.get('https://edition.cnn.com/search?q=&from=0&size=10&page=1&sort=newest&types=article&section=')
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_folder, f'cnn_news_{current_time}.csv')
    
    file_exists = os.path.isfile('cnn_news_original.csv')
    
    if file_exists:
        existing_data = pd.read_csv('cnn_news_original.csv')
        existing_titles_links = set(zip(existing_data['title'], existing_data['link']))
    else:
        existing_data = pd.DataFrame(columns=['title', 'link'])
        existing_titles_links = set()
    
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
>>>>>>> 460e1313f2de1733f91a03eb101dd2361a6e880d

        while True:
            try:
                class_elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 'container__item'))
                )
            except TimeoutException:
                print("無法載入頁面內容，結束程式。")
                break

<<<<<<< HEAD
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
                        items_list = {
                            'uuid': str(uuid.uuid4()),
=======
            for class_element in class_elements:
                try:
                    crawler_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    #print(crawler_datetime)
                    
                    unique_id = str(uuid.uuid4())
                    #print(unique_id)
                    
                    title = class_element.find_element(By.CLASS_NAME, 'container__headline-text').text
                    print(title)
                    
                    link = class_element.find_element(By.CLASS_NAME, 'container__link').get_attribute('href')
                    #print(link)
                    
                    if (title == last_title and link == last_link):
                        print("已經爬取過此文章，跳過此筆資料。")
                        continue
                    
                    if (title, link) not in existing_titles_links:
                        skip_count = 0
                        abstract = WebDriverWait(class_element, 3).until(
                                EC.presence_of_element_located((By.CLASS_NAME, 'container__description'))
                            ).text
                        #print(abstract)
                        
                        match = None
                        patterns = [
                            r"https://www\.cnn\.com/\d{4}/\d{2}/\d{2}/([^/]+)/",
                            r"https://www\.cnn\.com/cnn-underscored/([^/]+)/",
                            r'https?://www\.cnn\.com/([^/]+)/'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, link)
                            if match:
                                if len(match.group(1)) <= 25:
                                    category = match.group(1)
                                    any_category = "YES"
                                else:
                                    driver.execute_script(f"window.open('{link}', '_blank');")
                                    driver.switch_to.window(driver.window_handles[1])
                                    try:
                                        category_element = driver.find_element(By.CLASS_NAME, 'breadcrumb__link')
                                        category = category_element.text
                                        any_category = "YES"
                                    except:
                                        category = ""  
                                        any_category = "NO"
                                    driver.close()
                                    driver.switch_to.window(driver.window_handles[0])
                                    time.sleep(1)
                                break
                        else:
                            # 如果所有模式都沒有匹配成功
                            category = ""
                            any_category = "NO"
                                    
                            
                        #print(category)
                        #print(any_category)
                        
                        data_source = 'cnn_news'
                        #print(data_source)
                        
                        try:
                            date_str = class_element.find_element(By.CLASS_NAME, 'container__date').text
                            date_obj = datetime.strptime(date_str, "%b %d, %Y")
                            gattered_datetime = date_obj.strftime("%Y-%m-%d 00:00:00")
                            #print(gattered_datetime)
                            
                            if gattered_datetime.startswith("2023"):
                                print("資料日期為2023，停止爬取。")
                                driver.quit()
                                return filename
                            
                        except (NoSuchElementException, ValueError) as e:
                            if isinstance(e, NoSuchElementException):
                                print("日期元素不存在，跳過該項。")
                            else:
                                print(f"日期格式錯誤，無法解析日期: '{date_str}'")
                            continue  
                        
                        row_data = {
                            'uuid': unique_id,
>>>>>>> 460e1313f2de1733f91a03eb101dd2361a6e880d
                            'title': title,
                            'category': category,
                            'abstract': abstract,
                            'link': link,
<<<<<<< HEAD
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
                                writer.writerow(items_list)
                                
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
        
=======
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
                        
                        last_title, last_link = title, link
                        
                    else:
                        skip_count += 1  # 如果資料重複，計數器+1
                        print(f"資料已存在，跳過標題: {title}")
                        if skip_count >= 2:  # 檢查是否連續跳過兩筆
                            driver.quit()
                            return filename
                        
                except (NoSuchElementException, StaleElementReferenceException) as e:
                    print(f"遇到錯誤，跳過該項：{e}")
                    continue
                
            try:
                next_page_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="search"]/div[2]/div/div[4]/div/div[3]'))
                )
                next_page_button.click()
                time.sleep(random.uniform(2, 5)) 
            except (NoSuchElementException, TimeoutException):
                print("沒有找到下一頁，停止爬取。")
                break
            except ElementClickInterceptedException:
                print("無法點擊下一頁按鈕，可能是其他元素覆蓋了它。")
                break
            
>>>>>>> 460e1313f2de1733f91a03eb101dd2361a6e880d
    driver.quit()
    return filename 


