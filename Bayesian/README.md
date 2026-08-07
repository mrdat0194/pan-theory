# Solutions-to-Problems-in-Bayesian-Statistics
This repository contains my solutions to the assignments in the book: "A Student’s Guide to Bayesian Statistics" by Ben Lambert. I will update the repository with my solutions.

## Content

### An introduction to Bayesian inference

#### Chapter 2 - The subjective worlds of Frequentist and Bayesian statistics

#### Chapter 3 - Probability - the nuts and bolts of Bayesian inference

### Understanding the Bayesian formula

#### Chapter 4 - Likelihoods

#### Chapter 5 - Priors

#### Chapter 6 - The devil is in the denominator

#### Chapter 7 - The posterior - The goal of Bayesian inference

### Analytic Bayesian methods

#### Chapter 8 - Distributions

#### Chapter 9 - Conjugate priors

#### Chapter 10 - Evaluation of model fit and hypothesis testing

#### Chapter 11 - Making Bayesian analysis objective?

### Computational Bayes

#### Chapter 12 - Leaving conjugates behind: Markov chain Monte Carlo

#### Chapter 13 - Metropolis Hastings


##### Modeling presence of Borrelia amongst Ticks

###### Symmetric Kernel - Random Walk Metropolis
Using a Binomial likelihood, a Beta prior and an symmetric Normal jumping kernel.

###### Assymmetric Kernel - Metropolis Hastings
Using a Beta-Binomial likelihood, a Gamma prior and an assymmetric log-Normal jumping kernel.

##### Modeling Mosquito Death Rate
Using a Poisson Likelihood, a Gamma prior, a Beta Prior, a log-Normal jumping kernel and a beta jumping kernel. 
  
  
#### Chapter 14 - Gibbs Sampling

##### The sensitivity and specificity of a test for a disease - Gibbs Sampling

##### Coal mining disasters in the UK - Gibbs Sampling
Using Gibbs sampling to estimate the point in time when legislative and societal changes caused a reduction in coal mining disasters in
the UK. The number of disasters per year pre and post legislations were modeled using Poisson Likelihoods: Possion(lambda_1), Possion(lambda_2) with Gamma priors. The point in time when the new legislations were enacted is called n.

#### Chapter 15 - Hamiltonian Monte Carlo

#### Chapter 16 - Stan

### Hierarchical models and regression

#### Chapter 17 - Hierarchical models

#### Chapter 18 - Linear regression models

#### Chapter 19 - Generalized  linear models and other animals
### Computational Probability & Waiting Times

#### Unified Waiting-Time Distributions
We implement a unified framework in [`waiting_time_unification.py`](file:///c:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/waiting_time_unification.py) that covers the progression of discrete and continuous arrival metrics:
- **Geometric**: Discrete trial success count.
- **Exponential**: Continuous memoryless waiting time.
- **Erlang**: Multi-arrival Poisson processes.
- **Dirac Delta**: Deterministic limit (zero variance) modeled using a smoothed Gaussian kernel (matching the smoothed Dirac representation in [`importance_sampling_bayesian.py`](file:///c:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/importance_sampling_bayesian.py)).
- **Phase-Type**: Generalized continuous transitions via absorptive continuous-time Markov chains.







