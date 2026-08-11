---
name: convex-optimization-math
description: Activates whenever editing or creating machine learning solvers, linear models, ADMM, FISTA, custom gradient descent, or SVM/logistic regression models.
---
# Convex Optimization Mathematical Modeling

When working on ML solvers inside `MLModel/model/`:
- **Prefer Custom Solvers:** Use custom accelerated/proximal solvers (FISTA, ADMM) over generic `scikit-learn` black-box models.
- **Mathematical Rigor:** Emphasize convergence checks, Nesterov acceleration, and exact soft-thresholding (e.g., L1 regularization).
- **Interpretability:** Ensure the implementation remains fully interpretable with clear proximal operator steps.
