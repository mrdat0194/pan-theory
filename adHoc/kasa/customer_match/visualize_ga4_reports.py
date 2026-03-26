import os
import polars as pl
import matplotlib.pyplot as plt

# Revised visualization script with robust column identification and casting

def main():
    # 1. Setup Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    properties = ['vna', 'vinpearl', 'vinwonders']
    reports = ['traffic', 'pages', 'hardware', 'events', 'revenue']
    
    # 2. Create grid of plots (3 properties x 5 reports)
    fig, axes = plt.subplots(len(properties), len(reports), figsize=(30, 20))
    fig.suptitle('GA4 360 Properties: Top Data Visualization (Top 5 per Day - 15 Reports)', fontsize=26, y=0.98)
    
    cmap = plt.get_cmap('tab10')

    print("Generating visualization for 15 CSV reports...")

    for i, prop in enumerate(properties):
        for j, report in enumerate(reports):
            # Handle grid indexing
            if len(properties) == 1 and len(reports) == 1: ax = axes
            elif len(properties) == 1 or len(reports) == 1: ax = axes[max(i, j)]
            else: ax = axes[i, j]
                
            csv_file = f"{prop}_{report}_report.csv"
            csv_path = os.path.join(BASE_DIR, csv_file)
            
            if os.path.exists(csv_path):
                try:
                    # Read the CSV
                    df = pl.read_csv(csv_path)
                    
                    if df.is_empty():
                        ax.text(0.5, 0.5, 'Empty Data', ha='center')
                    else:
                        # --- ROBUST COLUMN IDENTIFICATION ---
                        # 1. 'date' is always col 0
                        # 2. Dimensions are usually the other String columns
                        # 3. Metrics are the Numeric columns
                        
                        # Identify numeric columns (metrics)
                        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Float64] and c != 'date']
                        
                        # If polars failed to infer types, force cast the last columns
                        if not numeric_cols:
                            # Try to cast everything after the first few cols to numeric
                            for c in df.columns[2:]:
                                try:
                                    df = df.with_columns(pl.col(c).cast(pl.Float64))
                                    numeric_cols.append(c)
                                except:
                                    pass
                        
                        if not numeric_cols:
                            ax.text(0.5, 0.5, 'No Metrics Found', ha='center')
                        else:
                            # Primary dimension: The first string column AFTER 'date'
                            string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 and c != 'date']
                            dim_col = string_cols[0] if string_cols else df.columns[min(1, len(df.columns)-1)]
                            metric_col = numeric_cols[0]
                            
                            # Find top 5 dimensions by volume
                            top_dims_df = df.group_by(dim_col).agg(pl.col(metric_col).sum()).sort(metric_col, descending=True).head(5)
                            top_dims = top_dims_df[dim_col].to_list()
                            
                            df = df.with_columns(pl.col('date').cast(pl.Utf8))
                            dates = sorted(df['date'].unique().to_list())
                            
                            for idx, dim_val in enumerate(top_dims):
                                dim_df = df.filter(pl.col(dim_col) == dim_val).group_by('date').agg(pl.col(metric_col).sum())
                                dates_df = pl.DataFrame({'date': dates})
                                dim_df = dates_df.join(dim_df, on='date', how='left').fill_null(0).sort('date')
                                
                                short_dim = str(dim_val)[:30] + '...' if len(str(dim_val)) > 30 else str(dim_val)
                                ax.plot(dates, dim_df[metric_col].to_list(), marker='o', label=short_dim, alpha=0.8, color=cmap(idx % 10))
                                
                            ax.set_title(f"{prop.upper()} - {report.capitalize()}", fontsize=14, fontweight='bold')
                            ax.tick_params(axis='x', rotation=45, labelsize=8)
                            ax.tick_params(axis='y', labelsize=10)
                            ax.set_ylabel(metric_col.replace('Count', ' Count'), fontsize=10)
                            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', fontsize=8, color='red')
            else:
                ax.text(0.5, 0.5, 'Missing CSV', ha='center', color='gray')
            
            ax.set_title(f"{prop.upper()} - {report.capitalize()}", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png = os.path.join(BASE_DIR, "ga4_multi_property_visual.png")
    plt.savefig(output_png, dpi=120)
    print(f"\nSUCCESS! Multi-report visualization saved to: {output_png}")

if __name__ == '__main__':
    main()
