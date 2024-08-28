from scipy.stats import norm
import numpy as np

# a = norm.cdf(350,250,75)
# a = norm.ppf(0.95,250,75)

# Yes4All Dropship sale volume of Premium Foam Roller follows a normal distribution
# with an average demand of 50 pcs and standard deviation of demand equal to 20 pcs.
# It takes 4 days for the local suppliers to deliver. The Dropship channel requires a service level of 95% (z = 1.64)
# and monitors its inventory continuously. Further consider inventory carrying cost per pcs per period to be $ 1.6
# and ordering cost per order to be $ 4000. Suppose the Supply Chain Mgr has adopted a
# periodic review policy to manage its inventory of Premium Foam Roller, and plans to place PO every five days.

day_to_check = 5
day_to_fill = 4
L = 4
Demand = 50
std_D = 20
Q = Demand*day_to_check
order_cost = 4000
holding_cost = 1.6

Average_Stock = Q/2

CR = 0.95

k = norm.ppf(CR)

h_y =0.6

safety_stock = norm.ppf(CR)*std_D*np.sqrt(day_to_check + day_to_fill)

order_upto = Demand*(day_to_check+day_to_fill) + k*std_D*np.sqrt(day_to_check+day_to_fill)

EOQ = np.sqrt((2*Demand*order_cost)/(holding_cost))

print(EOQ)

# not finished

from statsmodels.tsa.holtwinters import ExponentialSmoothing
# prepare data
data = [325, 340, 320, 350]
# create class
model = ExponentialSmoothing(data,trend = '0.1', damped_trend = '0.2' ,initial_trend= '20' )
print(model)