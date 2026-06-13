import os
import polars as pl
from GA4_CoreEngine import BuildReport, OrderBy

def main():
    # 1. Configuration for Vietnam Airlines GA4 (360 Property)
    PROPERTY_ID = '237200408' 
    
    # 2. Setup Credentials Path (Bubbly Service Account)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CREDS_PATH = os.path.join(BASE_DIR, 'bubbly-cascade-398303-5f3dd0a21703.json')
    
    if not os.path.exists(CREDS_PATH):
        print(f"Error: Credentials not found at {CREDS_PATH}")
        return

    # Dates to query individually to ensure we get "Top 50" for EACH day
    dates_to_query = ['today', 'yesterday', '2daysAgo']

    # 3. Define the Report Schemas (Adding 'date' dimension as requested)
    report_definitions = [
        {
            "file_tag": "traffic_source",
            "dimensions": ["date", "sessionSourceMedium"],
            "metrics": ["activeUsers", "sessions", "keyEvents"]
        },
        {
            "file_tag": "page_performance",
            "dimensions": ["date", "pagePath"],
            "metrics": ["screenPageViews", "activeUsers", "averageSessionDuration"]
        },
        {
            "file_tag": "platform_device",
            "dimensions": ["date", "deviceCategory", "operatingSystem"],
            "metrics": ["activeUsers", "sessions"]
        }
    ]

    print(f"Initializing Top 50 Per Day Report Export (last 3 days)...")
    
    try:
        for config in report_definitions:
            tag = config["file_tag"]
            first_metric = config["metrics"][0]
            
            print(f"\n--- Processing Report Task: {tag} ---")
            
            all_days_data = []

            for date_str in dates_to_query:
                print(f"  Fetching Top 50 samples for {date_str} (Sorted by {first_metric} DESC)...")
                
                # Using the BuildReport class from GA4_CoreEngine for a single day
                builder = BuildReport(
                    property_id=PROPERTY_ID,
                    ga_dimensions=config["dimensions"],
                    ga_metrics=config["metrics"],
                    start_date=date_str,
                    end_date=date_str,
                    creds_path=CREDS_PATH
                )
                
                # Define Sort Order: Descending by the first metric
                sort_order = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name=first_metric), desc=True)]
                
                # Fetch only 50 rows for this specific day
                df_day = builder.run_report(limit=50, order_bys=sort_order)
                all_days_data.append(df_day)
            
            # Combine all 3 days into one Polars DataFrame
            if all_days_data:
                final_df = pl.concat(all_days_data, how='vertical', rechunk=True)
                
                output_file = f"vna_ga4_{tag}_top50_per_day.csv"
                output_path = os.path.join(BASE_DIR, output_file)
                
                # Export to CSV
                final_df.write_csv(output_path)
                
                print(f"Successfully exported {len(final_df)} total rows to: {output_file}")
            else:
                print(f"No data found for report: {tag}")

        print("\nAll multi-day top-50 reports have been successfully exported.")

    except Exception as e:
        print(f"\nExecution Error: {e}")
        print("TIP: Ensure 'polars', 'numpy', and 'google-analytics-data' are installed.")

if __name__ == '__main__':
    main()
