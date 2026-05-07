import numpy as np
import pandas as pd
from scipy import stats

def one_sample_ztest():
    """
    Demonstrates a one-sample Z-test.
    Logic: Comparing a sample mean to a known population mean (mu0).
    """
    print("\n--- One-Sample Z-Test ---")
    np.random.seed(0)
    N = 100
    mu = 0.2
    sigma = 1
    x = np.random.randn(N) * sigma + mu
    
    mu0 = 0 # Null hypothesis mean
    
    # Calculation
    mu_hat = x.mean()
    # For Z-test, we usually assume population sigma is known or N is large
    # Here we use sample std dev as an estimate
    sigma_hat = x.std(ddof=1)
    
    z = (mu_hat - mu0) / (sigma_hat / np.sqrt(N))
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    
    print(f"Sample Mean: {mu_hat:.4f}")
    print(f"Z-statistic: {z:.4f}")
    print(f"P-value:     {p:.4f}")
    
    if p < 0.05:
        print("Result: Significant difference from mu0=0")
    else:
        print("Result: No significant difference")

def two_sample_ztest_proportions():
    """
    Demonstrates a two-sample Z-test for proportions using advertisement_clicks.csv.
    This is common in AB testing to compare conversion rates.
    """
    print("\n--- Two-Sample Z-Test for Proportions (AB Testing) ---")
    try:
        df = pd.read_csv('advertisement_clicks.csv')
        a = df[df['advertisement_id'] == 'A']['action']
        b = df[df['advertisement_id'] == 'B']['action']
        
        n1 = len(a)
        n2 = len(b)
        p1 = a.mean()
        p2 = b.mean()
        
        print(f"Ad A: n={n1}, conv_rate={p1:.4f}")
        print(f"Ad B: n={n2}, conv_rate={p2:.4f}")
        
        # Pooled proportion
        p_pooled = (a.sum() + b.sum()) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Z-statistic
        z = (p1 - p2) / se
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        
        print(f"Z-statistic: {z:.4f}")
        print(f"P-value:     {p:.4f}")
        
        if p < 0.05:
            print("Result: Significant difference between Ad A and Ad B")
        else:
            print("Result: No significant difference")
            
    except FileNotFoundError:
        print("Error: advertisement_clicks.csv not found.")

def titanic_fare_logic():
    """
    Downloads the Titanic dataset and performs a Z-test to compare
    the mean Fare of survivors vs non-survivors.
    """
    print("\n--- Titanic Fare Analysis (Full Implementation) ---")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    try:
        print(f"Loading data from {url}...")
        df = pd.read_csv(url)
        
        # Extract Fare data for survived (1) and not survived (0)
        x1 = df[df['Survived'] == 1]['Fare'].dropna().to_numpy()
        x0 = df[df['Survived'] == 0]['Fare'].dropna().to_numpy()
        
        # Sample sizes
        N1 = len(x1)
        N0 = len(x0)
        
        # Means
        mu1 = x1.mean()
        mu0 = x0.mean()
        
        # Variances (using ddof=1 for sample variance)
        s1 = x1.var(ddof=1)
        s0 = x0.var(ddof=1)
        
        # Standard error of the difference
        se = np.sqrt(s1 / N1 + s0 / N0)
        
        # Z-statistic (Null Hypothesis: mu1 - mu0 = 0)
        z = (mu1 - mu0) / se
        
        # P-value (two-tailed)
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        
        print(f"Survivors (N={N1}): Mean Fare = {mu1:.2f}")
        print(f"Non-Survivors (N={N0}): Mean Fare = {mu0:.2f}")
        print(f"Z-statistic: {z:.4f}")
        print(f"P-value:     {p:.4e}")
        
        if p < 0.05:
            print("Result: Reject Null Hypothesis. There is a significant difference in fares.")
        else:
            print("Result: Fail to reject Null Hypothesis.")
            
    except Exception as e:
        print(f"Error loading or processing Titanic data: {e}")

if __name__ == "__main__":
    one_sample_ztest()
    two_sample_ztest_proportions()
    titanic_fare_logic()
