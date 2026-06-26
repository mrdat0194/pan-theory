# from selenium.webdriver.chrome.service import Service
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import Select,WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver import ActionChains
# from selenium.webdriver.common.keys import Keys

import os
import pandas as pd
from main_def import MAIN_DIR
from pyppeteer import launch
import asyncio

def isNaN(num):
    return num != num

# Global Variable
save_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "info","url.csv")

async def capture(link, path_save):
    browser = await launch(headless=True, args=['--no-sandbox'])
    try:
        page = await browser.newPage()
        # Increased timeout to 60 seconds
        await page.goto(link, {'waitUntil': ['load', 'domcontentloaded', 'networkidle0', 'networkidle2'], 'timeout': 60000})
        await asyncio.sleep(8)
        await page.screenshot({'path': path_save, 'fullPage': True})
    finally:
        await browser.close()

async def process_link(link, n, semaphore):
    async with semaphore:
        print(f"Processing: {link}")
        path_save = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "Pic", str(n) + ".png")
        try:
            await capture(link, path_save)
        except Exception as e:
            print(f"Failed to capture {link}: {e}")
        await asyncio.sleep(1)

async def main():
    if not os.path.exists(save_path):
        print(f"File not found: {save_path}")
        return

    linkes = pd.read_csv(save_path)

    tasks = []
    semaphore = asyncio.Semaphore(2) # Limit concurrency to 2

    n = 0
    row_indexes = linkes.index
    for row in row_indexes:
        link = linkes['CaptureURL'].loc[row]
        if not isNaN(link):
            tasks.append(process_link(link, n, semaphore))
            n += 1

    if tasks:
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
