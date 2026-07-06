from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select,WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains

import time
from selenium.webdriver.common.keys import Keys
import pandas as pd
import requests
import pickle
import pyautogui
import pyperclip
import re
import os
import concurrent.futures

# Global Variable
driver = webdriver.Chrome(
    r"C:\Users\PeterN\PycharmProjects\pan-theory\Webscrape\chromedriver_win32\chromedriver.exe")

my_dict = {
    "txtName": [],
    "txtDuration": [],
    "txtFileLocation": []
}

def login(link):
    '''
    Login the website
    '''

    driver.get(link)

    element = driver.find_element_by_id('dnn_cmsHeader_inc_HomeMenuTop_MenuLoginMobile_lnkNotAuthen')

    driver.execute_script("arguments[0].click();", element)

    element = driver.find_element_by_id('dnn_cmsHeader_inc_HomeMenuTop_login_txtUsername')
    driver.execute_script('arguments[0].setAttribute("value","admin");', element)

    element = driver.find_element_by_id('dnn_cmsHeader_inc_HomeMenuTop_login_txtPassword')
    driver.execute_script('arguments[0].setAttribute("value","12345678");', element)

    select = Select(WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.ID, "dnn_cmsHeader_inc_HomeMenuTop_login_ddlUserType"))))
    select.options[1].click()

    element = driver.find_element_by_id('dnn_cmsHeader_inc_HomeMenuTop_login_lbtnLogin')
    driver.execute_script("arguments[0].click();", element)

def down_avatar(name, portal_link):
    '''
    Down Avatar trên Attech
    :param name:
    :param portal_link:
    :return:
    '''
    url = link_source + portal_link
    r = requests.get(url, allow_redirects=True)
    path_to_file = f"avatar/{name}.jpg"
    open(path_to_file, 'wb').write(r.content)

def them_lop_hoc(tenlop, info_name_edit,info_desc):
    '''
    Add class to the destination Web
    :param tenlop:
    :param info_name_edit:
    :param info_desc:
    :return:
    '''
    driver.get('https://daotao.baominh.vn/Quan-tri/quan-tri-khoa-hoc/khoa-hoc')

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_btnThem')
    driver.execute_script("arguments[0].click();", element)

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_lbtnAdd')
    driver.execute_script("arguments[0].click();", element)
    name_tenlop = f'arguments[0].setAttribute("value","{tenlop}");'
    time.sleep(1)
    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtName')
    driver.execute_script(name_tenlop, element)

    #Thêm avatar
    element = driver.find_element_by_xpath('//*[@id="collapse1"]/div/div[5]/div/button')
    element.click()
    tenlop_hinh = f"C:\\Users\\PeterN\\PycharmProjects\\pan-theory\\helper_functions\\auto_test_elearning\\avatar\\{info_name_edit}.jpg"
    time.sleep(1)
    # Tương tác với Window Folder Browser
    pyperclip.copy(tenlop_hinh)
    pyautogui.hotkey('right')
    pyautogui.hotkey("ctrl", "v")
    pyautogui.press('enter')
    time.sleep(10)

    #Thêm description
    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtDescription_contentIframe')
    driver.switch_to.frame(element)
    element = driver.find_element_by_xpath('/html/body')
    element.clear()
    element.send_keys(info_desc)
    time.sleep(10)
    driver.switch_to.default_content()

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtMaxUser')
    driver.execute_script('arguments[0].setAttribute("value","0");', element)

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtNgayBDDK')
    driver.execute_script('arguments[0].setAttribute("value","30/05/2021");', element)

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtNgayKTDK')
    driver.execute_script('arguments[0].setAttribute("value","30/05/2022");', element)

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtNgayBDH')
    driver.execute_script('arguments[0].setAttribute("value","30/05/2021");', element)

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtNgayKT')
    driver.execute_script('arguments[0].setAttribute("value","30/05/2022");', element)
    time.sleep(1)

    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_lbtnAddClass')
    driver.execute_script('arguments[0].click();', element)

    WebDriverWait(driver, 2).until(EC.alert_is_present())
    driver.switch_to.alert.accept()

    time.sleep(1)

def class_getinfo():
    '''
    Get Material from Attech from info: AIElearning_Class
    :return:
    '''
    baocao = pd.read_excel('baocaolophoc.xlsx', engine='openpyxl')
    row_indexes = baocao.index
    for row in row_indexes:
        info_name = baocao['Name'].loc[row]
        info_desc = baocao['Description'].loc[row]
        info_avatar = baocao['Avatar'].loc[row]
        info_name_edit = re.sub(r'[^a-zA-Z0-9 \n\.]', '', info_name)
        print(info_name_edit)
        try:
            down_avatar(info_name_edit ,info_avatar)
            them_lop_hoc(info_name,info_name_edit, info_desc)
        except:
            print(info_name)
            continue


def get_all_class():
    '''
    Because with just have 283 videos From HocLieu
    Already save in data_tailieu.pkl
    :return:
    '''
    for i in range(284):
         try:
            driver.get('http://attech.aivietnam.vn/Quan-tri/quan-tri-noi-dung/quantribaigiang')
            select = Select(WebDriverWait(driver, 1).until(
             EC.element_to_be_clickable((By.ID, "dnn_ctr437_inc_Main_ctl00_ddlType"))))
            select.options[2].click()
            select = Select(WebDriverWait(driver, 1).until(
             EC.element_to_be_clickable((By.ID, "dnn_ctr437_inc_Main_ctl00_ddlPageSize"))))
            select.options[7].click()
            print(f'//*[@id="tblObject"]/tbody/tr[{i+1}]/td[6]/div/button')
            element = driver.find_element_by_xpath(f'//*[@id="tblObject"]/tbody/tr[{i+2}]/td[6]/div/button')

            driver.execute_script('arguments[0].click();', element)
            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_rptMedia_lnkEdit_{i}')
            driver.execute_script('arguments[0].click();', element)
            time.sleep(5)
            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtName')
            my_dict['txtName'].append(element.get_attribute("value"))
            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtDuration')
            my_dict['txtDuration'].append(element.get_attribute("value"))
            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtFileLocation')
            my_dict['txtFileLocation'].append(element.get_attribute("value"))
         except:
            print(i)
            print(my_dict)
            a_file = open("data_tailieu.pkl", "wb")
            pickle.dump(my_dict, a_file)
            a_file.close()
            break

def _download_video_task(item):
    name, url, index = item
    try:
        info_name_edit = re.sub(r'[^a-zA-Z0-9 \n\.]', '', name)
        r = requests.get(url, timeout=30)
        path_to_file = f"down/{info_name_edit}.mp4"
        with open(path_to_file, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(f"down {index}: {e}")

def down_class():
    '''
    Down all the class, but not neccessary just down in Bulk:
    http://attech.aivietnam.vn/upload/145/145.zip
    :return:
    '''
    a_file = open("data_tailieu.pkl", "rb")
    my_dict = pickle.load(a_file)
    a_file.close()

    items_to_download = []
    for i in range(284):
        try:
            name = my_dict['txtName'][i]
            url = my_dict['txtFileLocation'][i]
            items_to_download.append((name, url, i))
        except:
            print("down prep", i)
            break
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(_download_video_task, items_to_download)

def get_and_down():
    '''
    Not Neccessary
    :return:
    '''
    get_all_class()
    down_class()


def add_beforeUpload():
    '''
    mydict_from data_tailieu.pkl input destination Web
    :return:
    '''
    a_file = open("data_tailieu.pkl", "rb")
    my_dict = pickle.load(a_file)
    a_file.close()
    for i in range(284):
        try:
            driver.get('https://daotao.baominh.vn/Quan-tri/quan-tri-noi-dung/quantribaigiang')

            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_lbtnAddVideo')
            driver.execute_script('arguments[0].click();', element)
            time.sleep(1)

            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_rddtCategory')
            driver.execute_script('arguments[0].click();', element)

            WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//*[@id="dnn_ctr437_inc_Main_ctl00_rddtCategory"]/span/span[1]'))).click()
            WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.XPATH,
                                            '//*[@id="dnn_ctr437_inc_Main_ctl00_rddtCategory_EmbeddedTree"]/ul/li/div/span'))).click()
            WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.XPATH,
                                            '//*[@id="dnn_ctr437_inc_Main_ctl00_rddtCategory_EmbeddedTree"]/ul/li/ul/li[2]/div/div/span'))).click()

            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtName')
            driver.execute_script(f'arguments[0].setAttribute("value", "{my_dict["txtName"][i]}");', element)
            print(my_dict["txtDuration"][i])
            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtDuration')
            driver.execute_script(f'arguments[0].setAttribute("value", "{my_dict["txtDuration"][i]}");', element)

            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtFileLocation')
            driver.execute_script(f'arguments[0].setAttribute("value", "{my_dict["txtFileLocation"][i]}");', element)

            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_lbtnUpdate')
            driver.execute_script('arguments[0].click();', element)

            WebDriverWait(driver, 2).until(EC.alert_is_present())
            driver.switch_to.alert.accept()

        except:
            print("down", i)
            break



def readCourse_classSource():
    '''
    Read and export cource from Attech:
    Already Save in folder lophocsource
    :return:
    '''
    for num in range(109):
        try:
            num_file = str(num)+'.xlsx'
            driver.get('http://attech.aivietnam.vn/Quan-tri/quan-tri-khoa-hoc/khoa-hoc')
            select = Select(WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.ID, "dnn_ctr457_inc_Main_ctl00_ddlPageSize1"))))
            select.options[7].click()
            element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_rptObject_LinkButton1_{num}')
            driver.execute_script('arguments[0].click();', element)
            element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnExport')
            driver.execute_script('arguments[0].click();', element)
            time.sleep(1)
            os.rename(r'C:\Users\PeterN\Downloads\baocao19.xlsx', r'C:\Users\PeterN\Downloads\%s'%num_file)
        except:
            print(num)

def hoclieu_scorm_video():
    '''
    Which class in Attech is Scorm and Which is Videos
    class 62 and 72 have neither
    :return:
    '''
    baocao = pd.read_excel('baocaolophoc.xlsx', engine='openpyxl')
    reversed_baocao = baocao.iloc[::-1]
    row_indexes = reversed_baocao.index
    reversed_baocao['typeClass'] = 0
    reversed_baocao['order'] = 0
    for num,index in enumerate(row_indexes):
        try:
            if num != 62 and num != 72:
                driver.get('http://attech.aivietnam.vn/Quan-tri/quan-tri-khoa-hoc/khoa-hoc')
                element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtkey')
                driver.execute_script(f'arguments[0].setAttribute("value", "{reversed_baocao["Name"].loc[index]}");',element)
                element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnSearch')
                driver.execute_script('arguments[0].click();', element)
                time.sleep(1)

                element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_rptObject_LinkButton1_0')
                driver.execute_script('arguments[0].click();', element)
                print(driver.current_url)
                if driver.current_url != 'http://attech.aivietnam.vn/ErrorPage/404.html':
                    try:
                        driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_rptObject_txtStartTime_0').text
                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_rptObject_lbtnEdit_0')
                        driver.execute_script('arguments[0].click();', element)
                    except:
                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_rptObject_lbtnEdit_1')
                        driver.execute_script('arguments[0].click();', element)


                    element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_rptData_lblType_0')

                    reversed_baocao['typeClass'].loc[index] = element.text
                    reversed_baocao['order'].loc[index] = num

        except:
            print(num)
            time.sleep(10000)
            reversed_baocao[['order','Name', 'typeClass']].to_excel("baocao.xlsx")

    reversed_baocao[['order','Name', 'typeClass']].to_excel("baocao.xlsx")




def addCourse_classDest():
    '''
    Add Course to Destination Web Class
    :return:
    '''
    pd.set_option("display.max_rows", None, "display.max_columns", 60, 'display.width', 1000)
    baocao = pd.read_excel('baocao.xlsx', engine='openpyxl')
    row_indexes = baocao.index

    for row in row_indexes:
        try:
            if baocao["typeClass"].loc[row] == 'Video':

                driver.get('https://daotao.baominh.vn/Quan-tri/quan-tri-khoa-hoc/khoa-hoc')
                element = driver.find_element_by_id('dnn_ctr457_inc_Main_ctl00_txtkey')
                driver.execute_script(f'arguments[0].setAttribute("value", "{baocao["Name"].loc[row]}");', element)
                element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnSearch')
                driver.execute_script('arguments[0].click();', element)
                time.sleep(1)
                element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_rptObject_LinkButton1_0')
                driver.execute_script('arguments[0].click();', element)
                lopcourse = pd.read_excel(f'lophocsource/{baocao["order"].loc[row]}.xlsx', engine='openpyxl')

                indexes = lopcourse.index
                for index in indexes:
                    if lopcourse['Loại dữ liệu'].loc[index] == 'Bài giảng':
                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnAddVideo')
                        driver.execute_script('arguments[0].click();', element)
                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnChooseMedia')
                        driver.execute_script('arguments[0].click();', element)
                        time.sleep(2)
                        print(lopcourse["Tên tài liệu"].loc[index])
                        element = driver.find_element_by_xpath(f'//*[@id="dnn_ctr457_inc_Main_ctl00_txtKeyword"]')
                        driver.execute_script(f'arguments[0].setAttribute("value", "{lopcourse["Tên tài liệu"].loc[index]}");', element)

                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnSearch')
                        driver.execute_script('arguments[0].click();', element)
                        time.sleep(1)
                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_rptMedia_lbtnChoose_0')
                        driver.execute_script('arguments[0].click();', element)
                        time.sleep(1)
                        element = driver.find_element_by_id(f'dnn_ctr457_inc_Main_ctl00_lbtnUpdate')
                        driver.execute_script('arguments[0].click();', element)
                        time.sleep(1)
        except:
            print(row)

def up_classHoclieu():
    a_file = open("data_tailieu.pkl", "rb")
    my_dict = pickle.load(a_file)
    a_file.close()

    for i in range(284):
        try:
            driver.get('https://daotao.baominh.vn/Quan-tri/quan-tri-noi-dung/quantribaigiang')

            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_txtKeyword')
            driver.execute_script(f'arguments[0].setAttribute("value", "{my_dict["txtName"][i]}");',
                                  element)

            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_lbtnSearch')
            driver.execute_script('arguments[0].click();', element)

            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_rptMedia_lnkEdit_0')
            driver.execute_script('arguments[0].click();', element)

            fix_name = my_dict["txtFileLocation"][i].replace("http://attech.aivietnam.vn/", "https://daotao.baominh.vn/")
            time.sleep(2)

            print(my_dict["txtName"][i])

            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtDuration')

            element.send_keys(my_dict["txtDuration"][i])
            pyperclip.copy(my_dict["txtDuration"][i])
            pyautogui.hotkey("ctrl", "v")

            element = driver.find_element_by_id('dnn_ctr437_inc_Main_ctl00_txtFileLocation')
            driver.execute_script(f'arguments[0].setAttribute("value", "{fix_name}");', element)

            time.sleep(1)
            element = driver.find_element_by_id(f'dnn_ctr437_inc_Main_ctl00_lbtnUpdate')
            driver.execute_script('arguments[0].click();', element)
            time.sleep(1)

            WebDriverWait(driver, 2).until(EC.alert_is_present())
            driver.switch_to.alert.accept()

        except:
            print("down", i)
            break


if __name__ == '__main__':
    # # info_tenlop = read_classInfo("baocao08.xlsx")
    link = 'https://daotao.baominh.vn/'
    link_source = 'http://attech.aivietnam.vn/'
    login(link)

    # login(link_source)
    # # class_getinfo()

    # get_and_down()

    # down_class()

    # a_file = open("data_tailieu.pkl", "rb")
    # my_dict = pickle.load(a_file)
    # a_file.close()
    #
    # print(len(my_dict['txtDuration']))
    #
    # add_beforeUpload()

    # readCourse_classSource()

    # hoclieu_scorm_video()

    # addCourse_classDest()

    up_classHoclieu()

    # driver.close()
    # driver.quit()

