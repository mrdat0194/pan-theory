# from selenium.webdriver.chrome.service import Service
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import Select,WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver import ActionChains
# from selenium.webdriver.common.keys import Keys

import os
import util
import time
import pandas as pd
from main_def import MAIN_DIR
from seleniumbase import SB

def isNaN(num):
    return num != num

executable_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "chromedriver_win32","chromedriver.exe")
print(executable_path)
# service = Service(executable_path=executable_path)
# options = webdriver.ChromeOptions()

# driver = webdriver.Chrome(service=service, options=options)

# Global Variable
save_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "info","url.csv")


from pyppeteer import launch
import asyncio


async def capture(link, path_save, ):
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(link, {'waitUntil': ['load', 'domcontentloaded', 'networkidle0', 'networkidle2']})
    await page.waitFor(8000)

    await page.screenshot({'path': path_save, 'fullPage': True})
    await browser.close()

linkes = pd.read_csv(save_path)
# print(linkes)

n = 0
row_indexes = linkes.index
for row in row_indexes:
    link = linkes['CaptureURL'].loc[row]
    if not isNaN(link):
        print(link)
        with SB(uc=True) as sb:
            sb.driver.get(link)
            time.sleep(2)
            path_save = os.path.join(MAIN_DIR,"ggl_api", "Automate_data_model","Pic", str(n) + ".png")
            # util.fullpage_screenshot(sb, path_save,link,row)
            asyncio.get_event_loop().run_until_complete(capture(link,path_save))

            time.sleep(1)
            n+=1