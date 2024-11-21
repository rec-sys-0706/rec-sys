from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import re
import csv
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import logging
import pandas as pd

class Article():
    def __init__(self, title=None, link=None, date=None, arxiv=None, abstract=None) -> None:
        self.title = title
        self.link = link
        self.date = date
        self.arxiv = arxiv
        self.abstract = abstract
    
    def __str__(self) -> str:
        return f'[{self.date}] {self.title}'

def get_daily_papers(data_dir=Path('data'), start_date='2023-05-04'):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir()
    
    driver = webdriver.Chrome()
    driver.get(f'https://huggingface.co/papers?date={start_date}')
    

    while True:
        baseURL = driver.current_url
        parsed_url = urlparse(driver.current_url)
        date = parse_qs(parsed_url.query)['date'][0]
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'section'))) # wait
        # ! NextURL
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # Scroll to bottom
        try:
            nextURL = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.LINK_TEXT, 'Next'))).get_attribute('href')
        except:
            nextURL = None

        filepath = data_dir / f'{date}.csv'
        logging.info(f'Processing papers on date [{date}]')
        if filepath.exists():
            logging.info('Data exists.')
        else:
            blocks = driver.find_elements(By.TAG_NAME, 'article')
            articles: list[Article] = []
            for block in blocks:
                h3 = block.find_element(By.TAG_NAME, 'h3')
                link = h3.find_element(By.TAG_NAME, 'a').get_attribute('href')
                if not link.startswith('https://huggingface.co/papers'):
                    raise ValueError(f'Link error. Got link: {link}')
                arxiv = 'https://arxiv.org/abs/' + re.sub('/papers/', '', urlparse(link).path)
                articles.append(Article(h3.text, link, date, arxiv))

            logging.info(f'{len(articles)} papers is under processing.')
            for article in articles:
                # Get Abstract
                driver.get(article.arxiv)
                abstract = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'abstract'))
                ).text
                article.abstract = abstract
            
                
            # Writing article attributes to a CSV file
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)                
                writer.writerow(['title', 'link', 'date', 'abstract']) # Header
                for article in articles:
                    writer.writerow([article.title, article.link, article.date, article.abstract])
                    print(article)

        if nextURL is None:
            break
        driver.get(nextURL)

    time.sleep(2)
    driver.quit()

def concat_papers(data_dir=Path('data')):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileExistsError(f"[ERROR] '{data_dir}' did not exist")
    csv_files = list(data_dir.glob('*.csv'))

    dfs = [pd.read_csv(file) for file in csv_files]

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df.to_csv(data_dir / 'combined.csv', index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    get_daily_papers('crawler/data')
    concat_papers('crawler/data')