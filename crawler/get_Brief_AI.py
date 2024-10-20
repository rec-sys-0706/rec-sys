from selenium import webdriver
import random
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import json
import requests
import os
from datetime import datetime

def scrape_brief_ai_news():
    
    with webdriver.Chrome() as driver:
        driver.get('https://ai.briefnewsletter.com/')

        homepage_link_list = []
        seen = set()
        
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

        class_elements = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'space-y-3'))
        )

        for class_element in class_elements:
            homepage_link = class_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
            if homepage_link not in seen:
                homepage_link_list.append(homepage_link)
                seen.add(homepage_link)

        print(homepage_link_list)

        items = []  

        for links in homepage_link_list:
            time.sleep(random.uniform(2, 5))
            try:
                driver.get(links)
                time.sleep(3)

                date_str_element = driver.find_element(By.CLASS_NAME, 'bh__byline_wrapper')
                date_str = date_str_element.find_element(By.CLASS_NAME, 'text-wt-text-on-background').text
                
                date_obj = datetime.strptime(date_str, "%B %d, %Y")
                gattered_datetime = date_obj.strftime("%Y-%m-%d 00:00:00")  

                sections = ['news', 'launch', 'research', 'insight']

                for section in sections:
                    try:
                        section_part = driver.find_element(By.ID, section)
                        title_element = section_part.find_element(By.TAG_NAME, 'h3')
                        title = title_element.find_element(By.CLASS_NAME, 'link').text or title_element.get_attribute('innerText')

                        paragraphs = section_part.find_elements(By.TAG_NAME, "p")
                        abstract_text = [
                            driver.execute_script("return arguments[0].innerText;", p) for p in paragraphs
                        ]
                        abstract = "\n".join(abstract_text)

                        link = title_element.find_element(By.CLASS_NAME, 'link').get_attribute('href')

                        if title and abstract and link and gattered_datetime:
                            item_data = {
                                'title': title,
                                'abstract': abstract,
                                'link': link,
                                'data_source': 'brief_ai_news',
                                'gattered_datetime': gattered_datetime
                            }
                            items.append(item_data)  

                            api_url = f"{os.environ.get('ROOT')}:5000/api/item/crawler"
                            if api_url:  
                                item_post = requests.post(api_url, json=item_data, timeout=20)
                                if item_post.status_code != 201:
                                    print(f"API 發送失敗: {item_post.text}")
                                if item_post.status_code == 201:
                                    print(f"API 發送成功: {item_post.text}")
                        else:
                            print(f"缺少資料: title={title}, abstract={abstract}, link={link}, gattered_datetime={gattered_datetime}")
                            
                    except NoSuchElementException:
                        print(f"在 {section} 中找不到所需元素")
                        continue

            except Exception as e:
                print(f"無法開啟 {links}: {e}")
                
        driver.quit()
        
        df = pd.DataFrame(items)
        df.to_csv('brief_ai_news.csv', index=False)

scrape_brief_ai_news()
