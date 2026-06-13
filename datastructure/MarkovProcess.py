from stochastic.processes.discrete import MarkovChain
import numpy as np

transition = np.array([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])

mar = MarkovChain(transition)
mar_sample = mar.sample(5)
print(mar_sample)