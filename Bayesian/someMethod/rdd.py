import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_rdd():
    """
    Demonstrates the Regression Discontinuity Design (RDD) causal inference method.
    """
    print("=== Regression Discontinuity Design (RDD) ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # Running variable (e.g., test score)
    x = np.random.uniform(0, 100, n)
    
    # Cutoff at 50
    cutoff = 50
    
    # Treatment assigned if x >= cutoff (Sharp RDD)
    treat = (x >= cutoff).astype(int)
    
    # Outcome variable
    # Baseline relationship: y = 10 + 0.5 * x
    # True treatment effect (jump at cutoff): 15
    y = 10 + 0.5 * x + 15 * treat + np.random.normal(0, 5, n)
    
    df = pd.DataFrame({'x': x, 'treat': treat, 'y': y})
    
    # Center the running variable around the cutoff for easier interpretation of the intercept
    df['x_centered'] = df['x'] - cutoff
    
    # Run RDD regression
    # We include an interaction term to allow for different slopes on either side of the cutoff
    model = smf.ols('y ~ x_centered * treat', data=df).fit()
    
    print(model.summary().tables[1])
    
    print("\nResults Analysis:")
    print("True Treatment Effect at Cutoff: 15.0")
    print(f"Estimated Treatment Effect (treat): {model.params['treat']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_rdd()
