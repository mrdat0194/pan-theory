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
plt.show()

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
plt.show()
