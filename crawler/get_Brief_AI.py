from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_brief_ai_news():
    driver = webdriver.Chrome()
    driver.get('https://ai.briefnewsletter.com/')

    homepage_link_list = []
    data = []
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

    for links in homepage_link_list:
        try:
            driver.get(links)
            time.sleep(3)  

            gattered_datetime_element = driver.find_element(By.CLASS_NAME, 'bh__byline_wrapper')
            gattered_datetime = gattered_datetime_element.find_element(By.CLASS_NAME, 'text-wt-text-on-background').text

            sections = ['news', 'launch', 'research', 'insight']

            for section in sections:
                try:
                    section_part = driver.find_element(By.ID, section)
                    title_element = section_part.find_element(By.TAG_NAME, 'h3')
                    title = title_element.find_element(By.CLASS_NAME, 'link').text

                    if not title:
                         title = title_element.get_attribute('innerText')

                    paragraphs = section_part.find_elements(By.TAG_NAME, "p")
                    abstract_text = [
                        driver.execute_script("return arguments[0].innerText;", p) for p in paragraphs
                    ]
                    abstract = "\n".join(abstract_text)

                    link = title_element.find_element(By.CLASS_NAME, 'link').get_attribute('href')

                    if title and abstract and link and gattered_datetime:
                        record = (title, abstract, link, gattered_datetime)
                        if record not in seen:
                            seen.add(record)
                            data.append({ 
                                'title': title,
                                'abstract': abstract,
                                'link': link,
                                'data_source': 'brief_ai_news',  
                                'gattered_datetime': gattered_datetime
                            })
                    else:
                        print(f"缺少資料: title={title}, abstract={abstract}, link={link}, gattered_datetime={gattered_datetime}")
                        
                except NoSuchElementException:
                    print(f"在 {section} 中找不到所需元素")
                    continue

        except Exception as e:
            print(f"無法開啟 {links}: {e}")

    driver.quit()
    
scrape_brief_ai_news()
