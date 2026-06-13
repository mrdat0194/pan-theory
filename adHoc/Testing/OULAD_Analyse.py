from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from helper_functions.OULAD_Helper import plotPerColumnDistribution
from helper_functions.OULAD_Helper import show_basic_info
import seaborn as sns
import matplotlib.pyplot as plt # plotting

print(os.listdir('../OULAD'))
nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../OULAD/assessments.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'assessments.csv'
# nRow, nCol = df1.shape
# print(f'There are {nRow} rows and {nCol} columns')
# print(df1.head(5))
# plotPerColumnDistribution(df1, 10, 5)

assessments_df = pd.read_csv('../OULAD/courses.csv')
# show_basic_info(assessments_df)

studentAssessment_df = pd.read_csv('../OULAD/studentAssessment.csv')
# show_basic_info(studentAssessment_df)

studentInfo_df = pd.read_csv('../OULAD/studentInfo.csv')
# show_basic_info(studentInfo_df)

studentRegistration_df = pd.read_csv('../OULAD/studentRegistration.csv')
# show_basic_info(studentRegistration_df)

studentVle_df = pd.read_csv('../OULAD/studentVle.csv')
# show_basic_info(studentVle_df)

vle_df = pd.read_csv('../OULAD/vle.csv')
# show_basic_info(vle_df)

#Checking gender distribution
sns.countplot(studentInfo_df.gender);    #this shows that courses data is almost equally distributed on gender

#Now let's try the same on age
studentInfo_df[['id_student', 'age_band']].groupby(by='age_band').count().plot.bar();    #this shows majority of students fall in age band of 0-35

#Now let's try the same on region
studentInfo_df[['id_student', 'region']].groupby(by='region').count().plot.bar();

# What if we want to do a multi dimensional visualization?
# Pandas provide this using crosstab
# crosstab: Compute a simple cross-tabulation of two (or more) factors. By default computes a frequency table of the factors
# unless an array of values and an aggregation function are passed.
pd.crosstab(studentInfo_df.region, studentInfo_df.age_band).plot.barh(stacked = True);

# How to visualize continous variables, ouliers?
# Python provides us boxplot for this.
# boxplot:  The box plot (a.k.a. box and whisker diagram) is a standardized way of displaying the
# distribution of data based on the five number summary:
# minimum, first quartile, median, third quartile, and maximum.
studentInfo_df.drop(['id_student', 'num_of_prev_attempts'], axis=1).boxplot(by = 'region')
plt.xticks(rotation = 90)    #without this, x-labels overlap




