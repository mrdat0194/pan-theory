# from scipy.stats import norm
# import numpy as np
#
# # a = norm.cdf(350,250,75)
# # a = norm.ppf(0.95,250,75)
#
# # Yes4All Dropship sale volume of Premium Foam Roller follows a normal distribution
# # with an average demand of 50 pcs and standard deviation of demand equal to 20 pcs.
# # It takes 4 days for the local suppliers to deliver. The Dropship channel requires a service level of 95% (z = 1.64)
# # and monitors its inventory continuously. Further consider inventory carrying cost per pcs per period to be $ 1.6
# # and ordering cost per order to be $ 4000. Suppose the Supply Chain Mgr has adopted a
# # periodic review policy to manage its inventory of Premium Foam Roller, and plans to place PO every five days.
#
# day_to_check = 5
# day_to_fill = 4
# L = 4
# Demand = 50
# std_D = 20
# Q = Demand*day_to_check
# order_cost = 4000
# holding_cost = 1.6
#
# Average_Stock = Q/2
#
# CR = 0.95
#
# k = norm.ppf(CR)
#
# # h_y = 0.6
#
# safety_stock = norm.ppf(CR)*std_D*np.sqrt(day_to_check + day_to_fill)
#
# print(safety_stock)
#
# order_upto = Demand*(day_to_check+day_to_fill) + k*std_D*np.sqrt(day_to_check+day_to_fill)
#
# print(order_upto)
#
# EOQ = np.sqrt((2*Demand*order_cost)/(holding_cost))
#
# print(EOQ)

# not finished

# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# # prepare data
# data = [325, 340, 320, 350]
# # create class
# model = ExponentialSmoothing(
#     data,
#     trend='additive',
#     damped_trend=True,
#     # damping_trend=0.2,  # This is the damping parameter
#     smoothing_level=0.1,  # This is the alpha parameter
#     initial_trend=float(20)
# )
# print(model)

# Dữ liệu ban đầu
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Các hàm riêng biệt
def is_even(n):
    return n % 2 == 0

def square(n):
    return n * n

# a = is_even(numbers)
# print(a)

# Thực hiện chuỗi xử lý dữ liệu
result = sum(map(square, filter(is_even, numbers)))
print(result) # Output: 220


# def outter(a):
#     var = "hi"
#     def inner(b):
#         print(a)
#         print(b)
#         print(var)
#
#     inside = inner("hello")
#
#     return inside
#
# outter("outside")
