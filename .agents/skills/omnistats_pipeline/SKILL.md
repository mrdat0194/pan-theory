---
name: omnistats-pipeline
description: Activates whenever the user asks to run, evaluate, plan, or edit experiments, perform power analysis, A/B testing, CUPED variance reduction, causal estimators, or generate APA reports.
---
# OmniStats Causal Inference and Experimentation Pipeline

When interacting with `omnistats/`, adhere to the 5-stage pipeline order:
1. **LPA baseline segmentation**
2. **Live test execution (Beta-Binomial)**
3. **CUPED variance reduction**
4. **Econometric causal evaluation (DiD, IV, RDD, SCM)**
5. **APA 7th report generation**

**Constraints:**
- PyMC NUTS sampling times can be long; exercise caution and use Importance Sampling fallbacks if requested.
- Prioritize Explainable AI (XAI) over black boxes.
