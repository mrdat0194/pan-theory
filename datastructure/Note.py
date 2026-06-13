#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:56:32 2019

@author: petern
"""

# No1 given the infinite sequence of numbers: {1,2,2,3,3,3,...} (number k apperas eactly k times in the sequence) given an integer i, return the ith digit in the sequence above

# def find_ith_digit(i):
#     # Find the group number to which the ith digit belongs
#     group = 1
#     while i > group * (group + 1) // 2:
#         group += 1
#
#     # Calculate the position of the ith digit within the group
#     position_within_group = i - group * (group - 1) // 2
#
#     # Return the group number (k) which represents the ith digit
#     return group
#
# # Example usage:
# i = 10
# result = find_ith_digit(i)
# print(f"The {i}th digit in the sequence is: {result}")


# No2 Huma khan is a graduate studnt at slide education. as a part of the regular assignment process, the students of slide education are asked to display the last unique character in the given string. help huma khan to solve the prolem in python. if no such unique character is present, display "none"
# def last_unique_character(s):
#     char_count = {}  # Dictionary to store character frequencies
#
#     # Count the occurrences of each character in the string
#     for char in s:
#         char_count[char] = char_count.get(char, 0) + 1
#
#     # Traverse the string in reverse to find the last unique character
#     for char in reversed(s):
#         if char_count[char] == 1:
#             return char
#
#     return "none"  # If no unique character is found
#
# # Example usage:
# input_string = "slideeducation"
# result = last_unique_character(input_string)
# print("The last unique character is:", result)

# No3 given a pyramid of integers, it is possible to traverse different paths from top to bottom. find the highest sum obtained by adding the numbers in the path using python
# def highest_sum_pyramid(pyramid):
#     if not pyramid or not pyramid[0]:
#         return 0
#
#     rows = len(pyramid)
#
#     # Create a memoization table to store the maximum sum for each position in the pyramid
#     memo = [[0] * (i + 1) for i in range(rows)]
#
#     # Initialize the memo table with the values of the bottom row of the pyramid
#     memo[rows - 1] = pyramid[rows - 1]
#
#     # Start from the second-to-last row and work upwards
#     for row in range(rows - 2, -1, -1):
#         for col in range(row + 1):
#             # The maximum sum at a position is the sum of the current value plus
#             # the maximum of the two adjacent values in the row below
#             memo[row][col] = pyramid[row][col] + max(memo[row + 1][col], memo[row + 1][col + 1])
#
#     # The highest sum will be at the top of the pyramid
#     return memo[0][0]
#
# # Example usage:
# pyramid = [
#     [1, 0, 0, 0],
#     [2, 3, 0, 0],
#     [3, 4, 6, 0],
#     [-1, -2, -3, -4]
# ]
#
# result = highest_sum_pyramid(pyramid)
# print("The highest sum obtained by traversing the pyramid is:", result)

#
# input_str = "1 0 0 0\n2 3 0 0\n3 4 6 0\n-1 -2 -3 -4"
#
# # Split the input string by newlines to get each row
# rows_str = input_str.split('\n')
#
# # Split each row string by spaces and convert elements to integers
# pyramid = [[int(num) for num in row_str.split()] for row_str in rows_str]
#
# # Print the resulting 2D list
# print(pyramid)



# Daily Coding Problem: Problem #687 [Hard]

# This problem was asked by Airbnb.
#
# An 8-puzzle is a game played on a 3 x 3 board of tiles, with the ninth tile missing. The remaining tiles are labeled 1 through 8 but shuffled randomly. Tiles may slide horizontally or vertically into an empty space, but may not be removed from the board.
#
# Design a class to represent the board, and find a series of steps to bring the board to the state [[1, 2, 3], [4, 5, 6], [7, 8, None]].

# #289
# The game of Nim is played as follows. Starting with three heaps, each containing a variable number of items, two players take turns removing one or more items from a single pile.
# https://github.com/ruppysuppy/Daily-Coding-Problem-Solutions/blob/master/Solutions/289.py

# #673
# This problem was asked by LinkedIn.
# 
# Given a list of points, a central point, and an integer k, find the nearest k points from the central point.
#
# For example, given the list of points [(0, 0), (5, 4), (3, 1)], the central point (1, 2), and k = 2, return [(0, 0), (3, 1)].



"""
#195
This problem was asked by Google.
Let A be an N by M matrix in which every row and every column is sorted.

Given i1, j1, i2, and j2, compute the number of elements of M smaller than M[i1, j1] and larger than M[i2, j2].

For example, given the following matrix:

[[1, 3, 7, 10, 15, 20],
 [2, 6, 9, 14, 22, 25],
 [3, 8, 10, 15, 25, 30],
 [10, 11, 12, 23, 30, 35],
 [20, 25, 30, 35, 40, 45]]
And i1 = 1, j1 = 1, i2 = 3, j2 = 3, return 15 as there are 15 numbers in the matrix smaller than 6 or greater than 23.

"""

# from typing import List
#
#
# def get_num_in_range(mat: List[List[int]], i1: int, j1: int, i2: int, j2: int) -> int:
#     num1, num2 = mat[i1][j1], mat[i2][j2]
#     count = sum([len([x for x in row if (x < num1 and x > num2)]) for row in mat])
#     return count
#
#
# if __name__ == "__main__":
#     mat = [
#         [1, 3, 7, 10, 15, 20],
#         [2, 6, 9, 14, 22, 25],
#         [3, 8, 10, 15, 25, 30],
#         [10, 11, 12, 23, 30, 35],
#         [20, 25, 30, 35, 40, 45],
#     ]
#     print(get_num_in_range(mat, 3, 3, 1, 1))
#
#     matrix = [
#         [1, 2, 3, 4],
#         [5, 8, 9, 13],
#         [6, 10, 12, 14],
#         [7, 11, 15, 16]
#     ]
#     print(get_num_in_range(matrix, 1, 3, 3, 1))
#
#     matrix = [
#         [1, 2, 3, 4],
#         [5, 6, 7, 8],
#         [10, 11, 12, 13],
#         [20, 21, 22, 23]
#     ]
#     print(get_num_in_range(matrix, 3, 3, 1, 0))
#
#
# """
# SPECS:
# TIME COMPLEXITY: O(n ^ 2)
# SPACE COMPLEXITY: O(n ^ 2)
# """

# from my_functions import timer

# 1. move_file

# from my_functions import Complete_Function
#
# Complete_Function.Move_file.Data_move("hello.py" ,"/Users/petern/Desktop/samV/Project/", "/Users/petern/Desktop/samV/Project/helper_functions/")

# 1. append to excel
# Complete_Function.append_df_to_excel("/Users/petern/Desktop/hello.xlsx" ,"a")

# Lesson 1

# from scipy.stats import norm
#
# print('%.4f' % (norm.cdf(-2, loc=3, scale=4) - norm.cdf(2, loc=3, scale=4)))
#
# print('%.4f' % (norm.cdf(-0.25) - norm.cdf(-1.25)))

# Lesson 2
# print("'HAN','SGN','DAD','PQC','CXR','BMV','CAH','VCA','VCL','VCS','DLI','DIN','VDH','HPH','HUI','PXU','UIH','VKG','THD','TBB','VDO','VII'".replace("'", "").replace(",","|"))

# Lesson 3

######
# setwd()
# Numpy
#
# import scipy.optimize
# import numpy as np
# def fun(y):
#     x= y**(1.0/3)+y-1
#     return x
#
# a = scipy.optimize.fsolve(fun, np.arange(0, 1, 0.01) )
# print(a[0])

######
# import numpy as np
# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# slice = arr[:2,1:3]
# print(slice)
# print(slice[0,0])

# print(arr[arr<0])
#
# A = np.array([[1],[2]])
# B = np.array([[1,2],[3,4]])
# print(A + B)

######
# Pandas
import pandas as pd

# Dict_to_frame
# Dictionary with list object in values
studentData = {
    'name': ['jack', 'Riti', 'Aadi'],
    'age': [34, 30, 16],
    'city': ['Sydney', 'Delhi', 'New york']
}

print('Creating Dataframe from Dictionary')

''' 
Pass dictionary in Dataframe constructor to create a new object
keys will be the column names and lists in values will be column data
'''
dfObj = pd.DataFrame(studentData)

# Print data frame object on console
print(dfObj)

print('Creating Dataframe from Dictionary and Custom Indexes')

# Pass custom names of index as list during initialization
dfObj = pd.DataFrame(studentData, index=['a', 'b', 'c'])

# Print dataframe object on console
print(dfObj)

print('Creating Dataframe from non compatible Dictionary')

studentAgeData = {
    'Jack': 12,
    'Roma': 13,
    'Ritika': 10,
    'Aadi': 11
}

'''
Creating dataframe by converting dict to list of items
'''
dfObj = pd.DataFrame(list(studentAgeData.items()), index=['a', 'b', 'c', 'd'])

# Print Dataframe object on console
print(dfObj)

print('Creating Dataframe from Dictionary by Skipping data')

studentData = {
    'name': ['jack', 'Riti', 'Aadi'],
    'age': [34, 30, 16],
    'city': ['Sydney', 'Delhi', 'New york']
}

# Creating Dataframe from Dictionary by Skipping 2nd Item from dict
dfObj = pd.DataFrame(studentData, columns=['name', 'city'])

# Print Dataframe object on console
print(dfObj)

print('Creating Dataframe from Dictionary With different orientation')

# Create dataframe from dic and make keys, index in dataframe
dfObj = pd.DataFrame.from_dict(studentData, orient='index')

print(dfObj)

print('Creating Dataframe from nested Dictionary')

# Nested Dictionary
studentData = {
    0: {
        'name': 'Aadi',
        'age': 16,
        'city': 'New york'
    },
    1: {
        'name': 'Jack',
        'age': 34,
        'city': 'Sydney'
    },
    2: {
        'name': 'Riti',
        'age': 30,
        'city': 'Delhi'
    }
}

'''
Create dataframe from nested dictionary 
'''
dfObj = pd.DataFrame(studentData)

# Print Dataframe object on console
print(dfObj)

print("Transpose the dictionary")

# Transpose dataframe object
dfObj = dfObj.transpose()

print(dfObj)

# Advanced CSV loading example
data = pd.read_csv(
"data/files/complex_data_example.tsv",      # relative python path to subdirectory
sep='\t' ,         # Tab-separated value file.
quotechar="'",        # single quote allowed as quote character
dtype={"salary": int},             # Parse the salary column as an integer
usecols=['name', 'birth_date', 'salary'] ,   # Only load the three columns specified.
parse_dates=['birth_date'],     # Intepret the birth_date column as a date
skiprows=10,         # Skip the first 10 rows of the file
na_values=['.', '??']  )     # Take any '.' or '??' values as NA

#files = {}
# 
#for filename in os.listdir("/Users/petern/Desktop/Python/Scrapy/data"):
#    if filename.endswith(".csv") and not filename in files:
##    os.path.isfile(filename) \
##       and
#        with open(filename, "r") as file:
#            files[filename] = file.read()            

#sys.path.append(os.getcwd())
# from module.file import MyClass
# instance = MyClass()

# REGX
#https://www.coursera.org/learn/python-network-data/supplement/2WnqH/python-regular-expression-quick-guide
#https://docs.python.org/3/howto/regex.html
import pandas as pd
data_file = pd.read_csv('2017-12-02-extract.csv', delimiter=',')

# Reverse a string

s = 'KDnuggets'

print('The reverse of KDnuggets is {}'.format(s[::-1]))

#

# Find character
import re
x = 'From: Using the : character'
y = re.findall('^F.+:', x)
print(y)

# Find all the number in text
import re
hand = open("datastructure/regex_sum.txt")
x=list()
for line in hand:
    y=re.findall('[0-9]+',line)
    x=x+y
sum=0
for i in x:
    sum=sum + int(i)
print(sum)

#

# Using urllib which is easier than socket
# socket is like a phonecall where you determine the connection
import urllib.request

fhand = urllib.request.urlopen('http://www.py4inf.com/code/romeo.txt')
print(type(fhand))
# for line in fhand:
#     print line.strip()

# You can count the words!

counts = dict()
for line in fhand:
    print(line.strip())
    words = line.split()
    for word in words:
        counts[word] = counts.get(word, 0) + 1
print(counts)

# Date time
import datetime
a = datetime.datetime(2019, 5, 26, 0, 0)
b =  datetime.datetime.today()

# dealing with str, replace, findall
# https://chrisalbon.com/python/data_wrangling/pandas_string_munging/

# merged = pd.merge(pop, abbrevs, how='outer',
#                  left_on='state/region', right_on='abbreviation')
# merged = merged.drop('abbreviation', 1) # drop duplicate info

# List

list = ['a']
list.index("a")
list.append("b")
list.insert(0,"c")
list.count("a")
print(list)
# Tuple is immutable

co = (4,5)
co[1]
print(co)

## dict

conversion = {
        "mar" : "March"
        }

conversion.get("jan")

conversion["mar"]

word ='brontosaurus'
d =dict()
for c in word:
    if c not in d:
        d[c] =1
    else:
        d[c] = d[c] +1
    print(d)

# Grid

abc = [ 
       [0,2,3],
       [1,2,3]       
       ]


# Math function
from math import *
ceil(4.76)


# COunt value, as dataframe

df['a'].value_counts()

df['freq'] = df.groupby('a')['a'].transform('count')

###

import pandas as pd
import dateutil
# Load data from csv file
data = pd.DataFrame.from_csv('/Users/petern/FirstPython/phone_data.csv')
# Convert date from string to date times
data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)



####  Neo4j
#$ cd /usr/local/Cellar/neo4j/
#$ cd {NEO4J_VERSION}/libexec/

#/usr/local/Cellar/neo4j/bin/neo4j-shell
#${NEO4J_ROOT}/bin/neo4j-shell


import re

str = 'thanhtaivtt'
match = re.match(r'[Tt]', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!')
else:
    print ('Khong tim thay!')
# Kết quả:
# Khop!
    
import re

str = 'thanhtaivtt'
match = re.match(r'[a-z0-9]', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!')
else:
    print ('Khong tim thay!')
# Kết quả:
# Khop!

#Ký tự dấu . này tương đương với việc so khớp một chuỗi phải chứa ít nhất một ký tự.


import re

str = 'thanhtaivtt'
match = re.match(r'.', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!')
else:
    print ('Khong tim thay!')
# Kết quả:
# Khop!  
    
#Ký tự này đại diện cho việc so khớp từ đầu của chuỗi.
#
#VD: so khớp xem có phải đầu chuỗi là chữ h

import re

str = 'thanhtaivtt'
match = re.search(r'^h', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!')
else:
    print ('Khong tim thay!')
# Kết quả:
# Khong tim thay!
#    
#Ký tự này đại diện cho việc so khớp đến cuối chuỗi.
#
#VD: so khớp xem cuỗi chuỗi có phải chữ t

import re

str = 'thanhtaivtt'
match = re.search(r't$', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong tim thay!')
# Kết quả:
# Khop!

#+
#Ký tự này đại diện cho có thể xuất hiện ít nhất hoặc nhiều ký tự trước nó.
#
#VD: So khớp xem một chuỗi chỉ chứa chữ cái in thường.

import re

str = 'vuthanhtai'
match = re.match(r'[a-z]+', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong tim thay!')
# Kết quả:
# Khop!
#    
#?
#Ký tự này đại diện cho chuỗi sẽ khớp với một trong các ký tự đằng trước nó.
#
#VD: Kiểm tra xem một chuỗi có được bắt đầu bằng a hoặc b hay không.

import re

str = 'vuthanhtai'
match = re.match(r'ab?', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong tim thay!')
# Kết quả:
# Khong tim thay!    

#{m, n}
#Ký tự này đại diện cho việc so khớp xem chuỗi đằng trước nó xuất hiện bao nhiêu tối thiểu m lần và tối đa n lần. Nếu bỏ trống n thì là so khớp sự xuất hiện m lần của chuỗi đằng trước nó.
#
#VD: Kiểm tra xem 3 chữ cái đầu của chuỗi có phải là v không.

import re

str = 'vuthanhtai'
match = re.match(r'v{3}', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong tim thay!')
# Kết quả:
# Khong tim thay!
    
#()
#
#Ký tự này dùng để gom nhóm các pattern lại với nhau.
#
#VD: Kiểm tra xem chuỗi chó chứa chũ 'thanh' không.

import re

str = 'vuthanhtai'
match = re.match(r'(.*)(thanh)(.*)', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong khop!')
# Kết quả:
# Khop!
    
#\
#
#Ký tự này giúp phân biệt chuỗi sau nó không phải là ký tự đặc biệt.
#
#VD: Kiểm tra xem đầu chuỗi có phải là dấu . không.

import re

str = '.vuthanhtai'
match = re.match(r'\.', str)
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong khop!')
# Kết quả:
# Khop!
    
#Pattern	Mô Tả
#\A	So khớp chuỗi là chuỗi.
#\B	So khớp nếu vị trí đặt \B không phải là đầu hoặc cuối chuỗi.
#\d	So khớp với số nguyên.
#\D	So khớp với các ký tự không phải là số.
#\s	So khớp với khoảng trắng và các ký tự chữ.
#\S	So khớp với các kỹ tự không phải là chữ.
#\w	So khớp với chữ hoặc số.
#\W	So khớp với các ký tự không phải là chữ hoặc số.
    
#Như ở phần 1 mình có trình bày các thông số flag trong Regular Expression, nhưng mình chưa liệt kê ra các flag đó. Và dưới đây là một số các flags hay dùng trong Python.
#
#I hoặc IGNORECASE - Không phân biệt hoa thường khi so khớp.
#L hoặc LOCALE - So Khớp với local hiện tại.
#M hoặc MULTILINE - Thay đổi $ và ^ thành kết thúc của một dòng và bắt đầu của một dòng thay vì mặc định là kết thúc chuỗi và bắt đầu chuỗi.
#A hoặc ACSII - Thay đổi \w, \W, \b, \B, \d, \D, \S và \s thành so khơp full unicode.
#S hoặc DOTALL -Thay đổi pattern . thành khớp với bất kỳ ký tự nào và dòng mới.
#.....
#VD: So khớp một chuỗi bắt đầu bằng v hoặc V.

import re

str = 'vuthanhtai'
match = re.match(r'V', str, re.IGNORECASE) # hoặc re.I
if match: #nếu tồn tại chuỗi khớp                     
    print('Khop!');
else:
    print ('Khong khop!')
# Kết quả:
# Khop!   

#http://emailregex.com/

string = "<red>AB CD<@red><yellow>EFGH<@yellow>"
#
data = re.findall(r'w*', string)
#['', 'red', '', 'AB', '', 'CD', '', '', 'red', '', '', 'yellow', '', 'EFGH', '', '', 'yellow', '', '']
data = re.findall(r'<w+>\w*<\w+>', string )
#['<yellow>EFGH<@yellow>']


# generator

from itertools import chain

list1 = ["a","b","c"]
list2 = ["d","e","f"]
list3 = ["g","h","i"]
print(list(chain(list1, list2, list3)))

iterables = [list1, list2, list3]
print(list(chain.from_iterable(iterables)))

def gen_iterables():
    for i in range(10):
        yield range(i)

print(list(chain.from_iterable(gen_iterables())))