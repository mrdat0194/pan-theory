# from selenium.webdriver.chrome.service import Service
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import Select,WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver import ActionChains
# from selenium.webdriver.common.keys import Keys

import os
import time
import pandas as pd
from main_def import MAIN_DIR


def isNaN(num):
    return num != num


# Global Variable
save_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "info", "url.csv")


from pyppeteer import launch
import asyncio


async def capture(
    link,
    path_save,
):
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(
        link,
        {"waitUntil": ["load", "domcontentloaded", "networkidle0", "networkidle2"]},
    )
    await page.waitFor(8000)

    await page.screenshot({"path": path_save, "fullPage": True})
    await browser.close()


async def main():
    linkes = pd.read_csv(save_path)
    # print(linkes)

    tasks = []
    # Using a semaphore to limit the number of concurrent browser instances
    semaphore = asyncio.Semaphore(3)

    async def process_link(row, n):
        async with semaphore:
            link = linkes["CaptureURL"].loc[row]
            if not isNaN(link):
                print(f"Capturing: {link}")
                # Removed redundant SeleniumBase (SB) call as it was blocking the event loop
                # and pyppeteer handles the capture independently.
                await asyncio.sleep(2)
                path_save = os.path.join(
                    MAIN_DIR, "ggl_api", "Automate_data_model", "Pic", str(n) + ".png"
                )
                # Capture the screenshot asynchronously
                await capture(link, path_save)
                await asyncio.sleep(1)

    for n, row in enumerate(linkes.index):
        tasks.append(process_link(row, n))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
