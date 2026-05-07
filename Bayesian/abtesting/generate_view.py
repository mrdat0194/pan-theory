import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# from scipy.stats import beta
#
# np.random.beta()

mu = 5
sigma2 = 1.3
N = 1000000
views = np.absolute(np.exp(stats.norm(mu, sigma2).rvs(N)).astype(np.int64) + 1)

p00 = np.percentile(views, 0)
p99 = np.percentile(views, 99)
ax = sns.histplot(views, bins=np.linspace(p00, p99, 50))
ax.set_title('Views, P99 = {}'.format(p99))
ax.set(xlabel = 'Views')
plt.tight_layout()
# plt.show() # Commented out to run headlessly without blocking

# expectation of our ground truth CTR, we will use it later to simulate B group
success_rate = 0.02
beta = 100
alpha = success_rate * beta / (1 - success_rate)
N = 1000000
success_rate = stats.beta(alpha, beta).rvs(N)

p00 = np.percentile(success_rate, 0)
p99 = np.percentile(success_rate, 99)
ax = sns.histplot(success_rate, bins=np.linspace(p00, p99, 50))
ax.set_title('CTR, P99 = {}'.format(p99))
ax.set(xlabel = 'CTR')
plt.tight_layout()
# plt.show() # Commented out to run headlessly without blocking

import pandas as pd
print("Generating clicks based on views and true CTR...")
# Simulate actual clicks using binomial distribution
# Each 'view' has a probability of 'success_rate' to be a click
clicks = np.random.binomial(views, success_rate)

df = pd.DataFrame({
    'views': views,
    'true_ctr': success_rate,
    'clicks': clicks
})

output_file = 'simulated_traffic.csv'
# Save a subset to keep the file size reasonable (100k rows)
print(f"Saving 100,000 rows to {output_file}...")
df.head(100000).to_csv(output_file, index=False)
print("Data saved successfully.")
