import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_did():
    """
    Demonstrates the Difference-in-Differences (DiD) causal inference method.
    """
    print("=== Difference-in-Differences (DiD) ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # 50% in treatment group
    treat = np.random.binomial(1, 0.5, n)
    
    # Pre-treatment period (post = 0) and post-treatment period (post = 1)
    # We create a panel dataset with 2 periods for each individual
    data = []
    for i in range(n):
        # Baseline outcome
        baseline = np.random.normal(10, 2)
        
        # Period 0 (Pre-treatment)
        y0 = baseline + np.random.normal(0, 1)
        data.append({'id': i, 'treat': treat[i], 'post': 0, 'y': y0})
        
        # Period 1 (Post-treatment)
        # True treatment effect is 5
        treatment_effect = 5 * treat[i]
        # Time trend (counterfactual change over time) is 2
        time_trend = 2
        y1 = baseline + time_trend + treatment_effect + np.random.normal(0, 1)
        data.append({'id': i, 'treat': treat[i], 'post': 1, 'y': y1})
        
    df = pd.DataFrame(data)
    
    # Run DiD regression
    # The interaction term 'treat:post' gives the causal effect estimator
    model = smf.ols('y ~ treat * post', data=df).fit()
    
    print(model.summary().tables[1])
    
    print("\nResults Analysis:")
    print("True Treatment Effect: 5.0")
    print(f"Estimated Treatment Effect (treat:post): {model.params['treat:post']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_did()
