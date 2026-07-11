import os
import sys
import pandas as pd

# Reconfigure stdout/stderr to UTF-8 to prevent UnicodeEncodeError in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Monkeypatch pandas_profiling import for QuickDA compatibility with pandas 2.0+ and numba 0.59+
try:
    import ydata_profiling
    sys.modules['pandas_profiling'] = ydata_profiling
except ImportError:
    pass

from quickda.clean_data import clean
from quickda.explore_data import explore

def main():
    print("--- Demo: QuickDA library ---")
    
    # Path to the downloaded dataset
    dataset_path = "titanic.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run demo_opendatasets.py first.")
        return
        
    print(f"Reading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print("\n1. Original Dataset Sample:")
    print(df.head())
    print(f"Original shape: {df.shape}")
    
    print("\n2. Data Cleaning with QuickDA (Standardizing Column names and dropping duplicates):")
    # clean() handles various standard cleaning tasks like duplicates, column headers, etc.
    df_clean = clean(df, method='default')
    print("Cleaned Columns:")
    print(df_clean.columns.tolist())
    
    print("\n3. Data Exploration - Generating Summary Stats Profile Report:")
    # Generates a quick HTML profile report named 'Titanic_QuickDA_Report.html'
    explore(df_clean, method='profile', report_name='Titanic_QuickDA_Report')
    print("Interactive HTML profile report saved as 'Titanic_QuickDA_Report.html'")

    print("\n4. Data Exploration - Generating Correlation Analysis:")
    # We select some numerical columns to explore correlations
    # In QuickDA, explore with method='correlation' plots/calculates correlation values
    numeric_cols = df_clean.select_dtypes(include=['number'])
    explore(numeric_cols, method='correlation')
    print("Correlation exploration completed successfully.")

if __name__ == "__main__":
    main()
