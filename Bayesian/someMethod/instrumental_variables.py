import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_iv():
    """
    Demonstrates the Instrumental Variables (IV) / 2-Stage Least Squares (2SLS) causal inference method.
    """
    print("=== Instrumental Variables (IV / 2SLS) ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # Unobserved confounder (causes endogeneity)
    u = np.random.normal(0, 2, n)
    
    # Instrument Z
    # Condition 1: Relevance (correlated with treatment X)
    # Condition 2: Exclusion restriction (uncorrelated with unobserved confounder U)
    z = np.random.normal(0, 2, n)
    
    # Endogenous treatment variable X
    # Affected by both Instrument Z and Confounder U
    x = 1.5 * z + 0.8 * u + np.random.normal(0, 1, n)
    
    # Outcome Y
    # True causal effect of X on Y is 2.0
    # Also affected by Confounder U, which creates omitted variable bias in standard OLS
    y = 5 + 2.0 * x + 1.5 * u + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({'y': y, 'x': x, 'z': z})
    
    # --- Naive OLS (Biased due to endogeneity) ---
    X_naive = sm.add_constant(df['x'])
    model_naive = sm.OLS(df['y'], X_naive).fit()
    print("--- Naive OLS Results (Biased) ---")
    print(f"Estimated Effect of X: {model_naive.params['x']:.4f}\n")
    
    # --- Two-Stage Least Squares (2SLS) ---
    
    # Stage 1: Regress endogenous X on instrument Z to isolate the exogenous variation
    Z_stage1 = sm.add_constant(df['z'])
    model_stage1 = sm.OLS(df['x'], Z_stage1).fit()
    df['x_hat'] = model_stage1.predict(Z_stage1)
    
    # Stage 2: Regress outcome Y on predicted X (x_hat) from Stage 1
    X_stage2 = sm.add_constant(df['x_hat'])
    model_stage2 = sm.OLS(df['y'], X_stage2).fit()
    
    print("--- Instrumental Variables (2SLS) Results ---")
    print(f"True Causal Effect of X: 2.0")
    print(f"Estimated Causal Effect (IV): {model_stage2.params['x_hat']:.4f}")
    
    print("\nNote: The standard errors from manual 2SLS stage 2 are slightly incorrect")
    print("because they don't account for the fact that x_hat is estimated.")
    print("For rigorous research, use a dedicated IV package like `linearmodels`:")
    print("`from linearmodels.iv import IV2SLS`")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_iv()
