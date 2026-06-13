import os
import polars as pl
from GA4_CoreEngine import BuildReport, OrderBy

def main():
    # 1. Properties to Analyze (3 major 360 accounts)
    properties = [
        {"id": "237200408", "name": "VNA"},
        {"id": "258003657", "name": "Vinpearl"},
        {"id": "318969518", "name": "VinWonders"}
    ]
    
    # 2. Setup Credentials Path (Bubbly Service Account)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CREDS_PATH = os.path.join(BASE_DIR, 'bubbly-cascade-398303-5f3dd0a21703.json')
    
    if not os.path.exists(CREDS_PATH):
        print(f"Error: Credentials not found at {CREDS_PATH}")
        return

    # Dates to query individually to ensure we get "Top 50" for EACH day
    # Changed to exclude today and get the last 3 full days
    dates_to_query = ['yesterday', '2daysAgo', '3daysAgo']

    # 3. Define the Report Types (4 per property)
    report_types = [
        {
            "tag": "traffic",
            "dimensions": ["date", "sessionSourceMedium"],
            "metrics": ["activeUsers", "sessions"]
        },
        {
            "tag": "pages",
            "dimensions": ["date", "pagePath"],
            "metrics": ["screenPageViews", "activeUsers"]
        },
        {
            "tag": "hardware",
            "dimensions": ["date", "deviceCategory", "operatingSystem"],
            "metrics": ["activeUsers"]
        },
        {
            "tag": "events",
            "dimensions": ["date", "eventName"],
            "metrics": ["eventCount", "activeUsers"]
        },
        {
            "tag": "revenue",
            "dimensions": ["date", "sessionSourceMedium"],
            "metrics": ["purchaseRevenue", "transactions"]
        }
    ]

    print(f"--- GA4 MASS DATA EXTRACTION (3 Properties x 4 Reports) ---")
    
    try:
        for prop in properties:
            p_id = prop["id"]
            p_name = prop["name"]
            print(f"\n>>>> PROPERTY: {p_name} ({p_id})")
            
            for config in report_types:
                tag = config["tag"]
                first_metric = config["metrics"][0]
                print(f"  Report: {tag} (Sorted by {first_metric} DESC)")
                
                all_days_data = []

                for date_str in dates_to_query:
                    # Initialize using GA4_CoreEngine
                    builder = BuildReport(
                        property_id=p_id,
                        ga_dimensions=config["dimensions"],
                        ga_metrics=config["metrics"],
                        start_date=date_str,
                        end_date=date_str,
                        creds_path=CREDS_PATH
                    )
                    
                    # Sort Descending by first metric
                    sort_order = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name=first_metric), desc=True)]
                    
                    # FETCH TOP 50 SAMPLES PER DAY
                    df_day = builder.run_report(limit=50, order_bys=sort_order)
                    all_days_data.append(df_day)
                
                # Combine days
                if all_days_data:
                    final_df = pl.concat(all_days_data, how='vertical', rechunk=True)
                    
                    # Final export name: [PROPERTY]_[TYPE]_report.csv
                    output_file = f"{p_name.lower()}_{tag}_report.csv"
                    output_path = os.path.join(BASE_DIR, output_file)
                    
                    final_df.write_csv(output_path)
                    print(f"    Exported {len(final_df)} rows to: {output_file}")

        print("\n--- ALL 15 REPORTS EXPORTED SUCCESSFULLY ---")

    except Exception as e:
        print(f"\nExecution Error: {e}")

if __name__ == '__main__':
    main()
