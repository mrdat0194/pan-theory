import pandas as pd
import numpy as np
from scipy import stats

def run_simulated_test():
    """
    Reads the simulated traffic data and performs a Z-test to check
    if the overall observed CTR matches the ground truth expectation.
    """
    print("--- Statistical Test on Simulated Data ---")
    try:
        df = pd.read_csv('simulated_traffic.csv')
    except FileNotFoundError:
        print("Error: simulated_traffic.csv not found. Please run generate_view.py first.")
        return

    # Calculate totals
    total_views = df['views'].sum()
    total_clicks = df['clicks'].sum()
    
    # Observed CTR
    observed_ctr = total_clicks / total_views if total_views > 0 else 0
    
    # Ground truth expectation from generate_view.py
    expected_ctr = 0.02
    
    print(f"Total Views:  {total_views:,}")
    print(f"Total Clicks: {total_clicks:,}")
    print(f"Observed CTR: {observed_ctr:.4f}")
    print(f"Expected CTR: {expected_ctr:.4f}")
    
def delta_method_test(df):
    """
    Performs a Z-test using the Delta Method to correctly estimate the 
    standard error of a ratio (clicks/views) for clustered user data.
    """
    print("\n--- Improved Test: The Delta Method ---")
    
    # X = Clicks per user, Y = Views per user
    X = df['clicks']
    Y = df['views']
    N = len(df)
    
    mu_x = X.mean()
    mu_y = Y.mean()
    var_x = X.var()
    var_y = Y.var()
    cov_xy = np.cov(X, Y)[0, 1]
    
    # Delta Method formula for the variance of the ratio (mu_x / mu_y)
    # Var(ratio) = (1/N) * [ (Var(X)/mu_y^2) + (mu_x^2 * Var(Y)/mu_y^4) - 2 * (mu_x * Cov(X,Y) / mu_y^3) ]
    var_ratio = (1/N) * (
        (var_x / (mu_y**2)) + 
        (mu_x**2 * var_y / (mu_y**4)) - 
        (2 * mu_x * cov_xy / (mu_y**3))
    )
    
    se_delta = np.sqrt(var_ratio)
    observed_ctr = mu_x / mu_y
    expected_ctr = 0.02
    
    z_stat = (observed_ctr - expected_ctr) / se_delta
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    print(f"Delta Method SE: {se_delta:.6f}")
    print(f"Z-statistic:    {z_stat:.4f}")
    print(f"P-value:        {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Reject Null Hypothesis (Difference is Significant)")
    else:
        print("Result: Fail to reject Null Hypothesis (Consistent with 0.02)")

if __name__ == "__main__":
    try:
        df_sim = pd.read_csv('simulated_traffic.csv')
        run_simulated_test() # Standard test
        delta_method_test(df_sim) # Improved test
    except FileNotFoundError:
        print("Error: simulated_traffic.csv not found. Run generate_view.py first.")
