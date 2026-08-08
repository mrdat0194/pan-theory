Read index.md instead, this is just teaching note. 

1) bai tap. mo folder bang commandline

2) excel : trình bày -> tìm cách : python : thực tiễn: implement

+ Đọc story_teller: biết thuật toán - data_structure

+ OOP : Object Oriented Programming - code SOLID 
  
+ Phân biệt: imperative # functional programming

+ tạo def temp_close_0 trong story_teller

+ summary: https://goodresearch.dev/decoupled.html

3) subscribe: https://www.dailycodingproblem.com/
+ sort and timing

4) Thuat toan:
+   https://github.com/TheAlgorithms/Python
+    https://github.com/keon/algorithms.git
   
   
5) Solid coding/ Clean code:
+ https://github.com/PacktPublishing/Clean-Code-in-Python-Second-Edition.git
+ https://github.com/mynameisfiber/high_performance_python_2e.git


6) ML/AI: 
+ https://probml.github.io/pml-book/
+ https://github.com/Atcold/pytorch-Deep-Learning.git
+ https://github.com/eriklindernoren/ML-From-Scratch.git
+ https://github.com/ageron/handson-ml2.git


Note: Do not duplicate story or lesson. Created.

---

### Class Imbalance Mitigation Notes
* **Imbalance Sweep Improvements**: Applying SMOTEENN to Random Forest increased the F1-Score from **0.2326** to **0.7119** (a **+206%** improvement). Detailed sweep results can be found in [MLModel/run/README.md](file:///c:/Users/mrdat/PycharmProjects/pan-theory/MLModel/run/README.md).

---

### Stage-by-Stage Causal & Experimentation Pipeline: OmniStats (v3)
The project now includes **OmniStats** under [omnistats/](file:///c:/Users/mrdat/PycharmProjects/pan-theory/omnistats/), a production-grade 5-stage causal inference and sequential experimentation pipeline:
1. **Stage 1 — Latent Profile Analysis (LPA)**: Describes the population by segmenting users into Gaussian Mixture Model classes. Outputs `profile_prob_max` for variance reduction.
2. **Stage 2 — Sequential Bayesian A/B Testing**: Compares variants using Beta-Binomial conjugate updates and StudentT means models running on PyMC NUTS (with Importance Sampling fallback). Decides to stop using Expected Loss.
3. **Stage 3 — CUPED Variance Reduction**: Sharpens subsequent estimations using monotonic regression (CatBoost/DecisionTree) on the LPA posterior class probabilities as pre-experiment covariates.
4. **Stage 4 — Causal Inference Suite**: Attributes causation using econometric estimators operating on CUPED-adjusted outcome data:
   * Staggered DiD (Callaway & Sant'Anna)
   * Instrumental Variables (linearmodels 2SLS)
   * Regression Discontinuity Design (rdrobust CCT)
   * Synthetic Control Method (Adaptive Proximal Gradient weight optimization with Scipy SLSQP fallback)
   * Matrix Completion (SoftImpute Nuclear Norm regularization)
   * Bayesian Structural Time Series (Google CausalImpact BSTS with spike-and-slab selection)
5. **Stage 5 — APA Report Consolidation**: Assembles all findings into an APA 7th edition Word document (Tables 1-8).

* **Explainable AI (XAI)**: The pipeline rejects opaque black boxes in favor of interpretable econometric models (SCM donor weights, CausalImpact control series inclusion probability, and Bayesian credible intervals).


---

### MLModel: State-of-the-Art (SOTA) Convex Optimization
The machine learning models in `MLModel/model/` have been upgraded from standard `scikit-learn` black-box solvers to custom-built, mathematically rigorous **Convex Optimization** algorithms (inspired by Stephen Becker's optimization literature):
1. **FISTA (Accelerated Proximal Gradient):** Replaces standard Logistic Regression. Uses Nesterov acceleration and exact soft-thresholding to optimize $L_1$-regularized Sparse Logistic Regression at $O(1/k^2)$ convergence. See [fista_logistic.py](file:///c:/Users/mrdat/PycharmProjects/pan-theory/MLModel/model/fista_logistic.py).
2. **ADMM (Alternating Direction Method of Multipliers):** Replaces `LinearSVC`. Accurately resolves the non-differentiable Hinge Loss in Support Vector Machines using custom asymmetric proximal operators. See [admm_svm.py](file:///c:/Users/mrdat/PycharmProjects/pan-theory/MLModel/model/admm_svm.py).

These solvers ensure full white-box interpretability and mathematical control over regularization constraints.
