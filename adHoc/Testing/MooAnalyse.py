import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_rows", None, "display.max_columns", 80, 'display.width', 1000)
from helper_functions.OULAD_Helper import show_basic_info

# print(os.listdir("Export2016/export/"))
#
# mdl_user = pd.read_csv('Export2016/export/mdl_user.csv')
# mdl_user.dataframeName = 'mdl_user.csv'
#
# mdl_badge_issued = pd.read_csv('Export2016/export/mdl_badge_issued.csv')
# mdl_badge_issued.dataframeName = 'mdl_badge_issued.csv'
#
# mdl_course_modules = pd.read_csv('Export2016/export/mdl_course_modules.csv')
# mdl_course_modules.dataframeName = 'mdl_course_modules.csv'
#
# mdl_course_modules_completion = pd.read_csv('Export2016/export/mdl_course_modules_completion.csv')
# mdl_course_modules_completion.dataframeName = 'mdl_course_modules_completion.csv'
#
# mdl_grade_grades_history = pd.read_csv('Export2016/export/mdl_grade_grades_history.csv')
# mdl_grade_grades_history.dataframeName = 'mdl_grade_grades_history.csv'
#
# mdl_logstore_standard_log = pd.read_csv('Export2016/export/mdl_logstore_standard_log.csv')
# mdl_logstore_standard_log.dataframeName = 'mdl_logstore_standard_log.csv'
#
#
# from helper_functions.OULAD_Helper import plotPerColumnDistribution
# # plotPerColumnDistribution(mdl_grade_grades_history, 9, 7)
#
# show_basic_info(mdl_grade_grades_history)

from ggplot import *
# Draw a stacked bar chart
# Cannot be done: pd.crosstab(grouped_clicks.code_module,[grouped_clicks.click_per_person, grouped_clicks.activity_type]).plot.bar(stacked = True);
grouped_clicks = pd.read_csv("../grouped_click.csv")
ggplot(grouped_clicks,aes( y="click_per_person", x="code_module", fill="activity_type")) + geom_bar( position = 'stack',stat='identity')