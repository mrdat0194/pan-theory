# chisqtest: 2 things is not related

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

# contingency table
#        click       no click
#------------------------------
# ad A |   a            b
# ad B |   c            d


def get_p_value(T):
  # same as scipy.stats.chi2_contingency(T, correction=False)
  det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
  c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
  p = 1 - chi2.cdf(x=c2, df=1)
  return p

# get data
df = pd.read_csv('advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action']
b = b['action']

A_clk = a.sum()
A_noclk = a.size - a.sum()
B_clk = b.sum()
B_noclk = b.size - b.sum()

T = np.array([[A_clk, A_noclk], [B_clk, B_noclk]])
# Most authors refer to statistically significant as P < 0.05 and statistically highly significant as P < 0.001 (less than one in a thousand chance of being wrong)
print(T)
print(get_p_value(T))

# This test show ab test where a is better than b or not when both is statiscally independent.
def bayes_ab_test(df, seed=1984):
    np.random.seed(seed)
    clicks = df
    exposure_a = len(clicks[clicks['advertisement_id'] == "A"])
    exposure_b = len(clicks[clicks['advertisement_id'] == "B"])
    click_a = len(clicks[clicks['advertisement_id'] == "A"][clicks['action'] == 1])
    click_b = len(clicks[clicks['advertisement_id'] == "B"][clicks['action'] == 1])
    post_a = np.random.beta(1 + click_a, 5 + exposure_a - click_b, 10000)
    post_b = np.random.beta(1 + click_b, 5 + exposure_b - click_b, 10000)

    prob_a_better = (post_a > post_b).mean()
    return prob_a_better

result = bayes_ab_test(df)
print(result)

df.plot()
