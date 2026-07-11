from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome("C:\Hello\AI\Webscrape\chromedriver_win32\chromedriver.exe")

driver.get('http://vietnamonlineacademy.vn/Account/Login')

dataSet = []

element = driver.find_element_by_id('UsernameOrEmailAddress')
driver.execute_script('arguments[0].setAttribute("value","admin");', element)
#
# element = driver.find_element_by_id('dnn_cmsHeader_inc_HomeMenuTop_login_txtPassword')
# driver.execute_script('arguments[0].setAttribute("value","12345678");', element)
#
# element = driver.find_element_by_id('dnn_cmsHeader_inc_HomeMenuTop_login_lbtnLogin')
# driver.execute_script("arguments[0].click();", element)
#
# # Click vào khung manager góc phải
# element = driver.find_element_by_id('dnn_ctr490_inc_Main_ctl00_divShow')
# driver.execute_script("arguments[0].click();", element)
# # # Href bài giảng
# # element = driver.find_elements_by_css_selector(".text-center [href]")
# # links = [elem.get_attribute('href') for elem in element]
# # driver.get(links[0])
#
# # Quản trị đào tạo
# element = driver.find_elements_by_css_selector(".listmenumobileuser [href]")
# links = [elem.get_attribute('href') for elem in element]
# driver.get(links[0])



# elems = WebDriverWait(driver,10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".text-center [href]")))
# print(elems)
#
# elems = WebDriverWait(driver,2).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".btn btn-default [href]")))
# print(elems)

# driver.execute_script("arguments[0].click();", element)

# totalBooks = driver.find_element_by_xpath("//*[@id='default']//form/strong[1]")
# print("Found: ", totalBooks.text)
#
# page = True
# while page:
#     listings = driver.find_elements_by_xpath("//*[@id='default']//ol/li[position()>0]")
#     for listing in listings:
#         url = listing.find_element_by_xpath(".//article[contains(@class,'product_pod')]/h3/a").get_attribute('href')
#         title = listing.find_element_by_xpath(".//article[contains(@class,'product_pod')]/h3/a").text
#         titleLarge = listing.find_element_by_xpath(".//article[contains(@class,'product_pod')]/h3/a").get_attribute(
#             'title')
#         price = listing.find_element_by_xpath(".//article/div[2]/p[contains(@class,'price_color')]").text
#         stock = listing.find_element_by_xpath(".//article/div[2]/p[2][contains(@class,'availability')]").text
#         image = listing.find_element_by_xpath(
#             ".//article/div[1][contains(@class,'image_container')]/a/img").get_attribute('src')
#         starRating = listing.find_element_by_xpath(".//article/p[contains(@class,'star-rating')]").get_attribute(
#             'class')
#         dataSet.append([titleLarge, title, price, stock, image, starRating.replace('star-rating ', ''), url])
#
#     try:
#         #Check for Pagination with text 'next'
#         driver.find_element_by_link_text('next').click()
#         continue
#     except NoSuchElementException:
#         page = False
#
# print("Completed")
#
# print(dataSet)
#
# driver.close()
# driver.quit()
