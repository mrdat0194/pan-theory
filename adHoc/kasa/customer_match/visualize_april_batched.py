import os
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Visualization script for April 2026 Batched GA4 Reports

def main():
    # 1. Setup Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Property names as they appear in the filenames: april_2026_{prop.lower()}_{tag}_full.csv
    properties = ['vna', 'vinpearl', 'vinwonders']
    reports = ['traffic', 'pages', 'hardware', 'events', 'revenue']
    
    # 2. Create grid of plots (3 properties x 5 reports)
    fig, axes = plt.subplots(len(properties), len(reports), figsize=(32, 22))
    fig.suptitle('GA4 April 2026: Batched Data Visualization (Top 7 per Category)', fontsize=28, y=0.98)
    
    cmap = plt.get_cmap('tab10')

    print("Generating visualization for April 2026 batched reports...")

    for i, prop in enumerate(properties):
        for j, report in enumerate(reports):
            # Handle grid indexing
            ax = axes[i, j]
                
            csv_file = f"april_2026_{prop}_{report}_full.csv"
            csv_path = os.path.join(BASE_DIR, csv_file)
            
            if os.path.exists(csv_path):
                print(f"  Processing: {csv_file}")
                try:
                    # Read the CSV
                    df = pl.read_csv(csv_path)
                    
                    if df.is_empty():
                        ax.text(0.5, 0.5, 'Empty Data', ha='center', transform=ax.transAxes)
                    else:
                        # --- DATA CLEANING & PREPARATION ---
                        # 1. Handle the 'date' column (YYYYMMDD to Date object)
                        df = df.with_columns(
                            pl.col('date').cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d")
                        )
                        
                        # 2. Robust Column Identification
                        # Metrics: Numeric columns (excluding date which was just converted)
                        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int32, pl.Int64, pl.Float64] and c != 'date']
                        
                        # Dimensions: First string column (excluding date)
                        string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 and c != 'date']
                        dim_col = string_cols[0] if string_cols else None
                        
                        if not numeric_cols or not dim_col:
                            ax.text(0.5, 0.5, 'Incomplete Data structure', ha='center', transform=ax.transAxes)
                        else:
                            metric_col = numeric_cols[0] # Use the first metric for visualization
                            
                            # Find top 7 dimension values by total metric sum
                            top_dims_df = df.group_by(dim_col).agg(pl.col(metric_col).sum()).sort(metric_col, descending=True).head(7)
                            top_dims = top_dims_df[dim_col].to_list()
                            
                            # Get all unique dates for consistent plotting
                            all_dates = sorted(df['date'].unique().to_list())
                            
                            for idx, dim_val in enumerate(top_dims):
                                # Filter for this dimension and aggregate by date
                                dim_df = df.filter(pl.col(dim_col) == dim_val).group_by('date').agg(pl.col(metric_col).sum())
                                
                                # Ensure we have data for all dates (fill with zeros)
                                dates_df = pl.DataFrame({'date': all_dates})
                                dim_df = dates_df.join(dim_df, on='date', how='left').fill_null(0).sort('date')
                                
                                # Shorten long dimension names
                                short_dim = str(dim_val)[:25] + '..' if len(str(dim_val)) > 25 else str(dim_val)
                                
                                ax.plot(dim_df['date'].to_list(), dim_df[metric_col].to_list(), 
                                        marker='o', markersize=4, label=short_dim, alpha=0.8, color=cmap(idx % 10))
                                
                            # Formatting
                            ax.set_title(f"{prop.upper()} - {report.capitalize()}", fontsize=15, fontweight='bold')
                            
                            # X-Axis formatting (Dates)
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5)) # Label every 5 days
                            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                            
                            ax.tick_params(axis='y', labelsize=10)
                            ax.set_ylabel(metric_col, fontsize=10)
                            
                            # Legend on the right side if it fits, otherwise below
                            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
                            
                except Exception as e:
                    print(f"    Error processing {csv_file}: {e}")
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:40]}', ha='center', transform=ax.transAxes, fontsize=8, color='red')
            else:
                ax.text(0.5, 0.5, 'Missing CSV', ha='center', transform=ax.transAxes, color='gray')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png = os.path.join(BASE_DIR, "ga4_april_batched_visual.png")
    plt.savefig(output_png, dpi=130)
    print(f"\nSUCCESS! April 2026 visualization saved to: {output_png}")

if __name__ == '__main__':
    main()
