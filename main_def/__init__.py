import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
query_path = os.path.join(MAIN_DIR, "info", "query.txt")
data_trial = os.path.join(MAIN_DIR, "data", "Gonj")
config_sql = os.path.join(MAIN_DIR, "sql_con", "database.json")
credentials = os.path.join(MAIN_DIR, "ggl_api", "kasa", "GoogleAds", "credentials.json")
tokens = os.path.join(MAIN_DIR, "ggl_api", "token.pickle")

# function_utils = os.path.join(MAIN_DIR,"Complete_Function.py")
# exec(open(function_utils).read())
#
# import os
# import csv
# MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
# credentials = os.path.join(MAIN_DIR, "ggl_api", "credentials.json")
# client_secret_store = os.path.join(MAIN_DIR, "ggl_api", "client_secret_store.json")
# tokens = os.path.join(MAIN_DIR, "ggl_api", "token.pickle")
#
# save_path_sheet = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "info", "to_sheet.csv")
# get_url = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "info","url.csv")
# image_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "Pic")
#
# # Change path or delete files in drive /Pic target_folder_id
# target_folder_id = '1TTpwAAzmBTLKDOtPjETKFOQAyzHAN_wW'
# slide_image_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "info", "slide_image_path.csv")
#
# # Global variable
# FIX_SLIDE_ID = "1G8CSwm5UhjXpwqBmr5SdfMtME_5KvZ4bT6_BbMaanPY" # Template dev review
#
# to_sheet = []
# with open(save_path_sheet, newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in reader:
#         print(', '.join(row))
#         to_sheet.append(row[0])
#
# # Input
# title_sheet = "[Yapo] dev review - data model - dev guide"
# title_test_plan = "[Yapo] Test plan"
#
# # Change name of slide
# new_slide = "[Yapo] - Web Dev Review 2023 v0.0"
#
# title_sheet = "[Yapo] dev review - data model - dev guide"


