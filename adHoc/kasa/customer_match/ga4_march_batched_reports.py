import os
import polars as pl
import time
from datetime import datetime, timedelta
from GA4_CoreEngine import BuildReport, OrderBy

def get_date_batches(start_date_str, end_date_str, batch_days=3):
    """Generates (start, end) date strings for batches."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    batches = []
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + timedelta(days=batch_days - 1)
        if current_end > end_date:
            current_end = end_date
        
        batches.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end + timedelta(days=1)
    
    return batches

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

    # Generate March 2026 Batches (3 days each)
    march_batches = get_date_batches("2026-03-01", "2026-03-31", batch_days=3)

    # 3. Define the Report Types (5 per property)
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

    print(f"--- GA4 MARCH DATA EXTRACTION (3 Properties x 5 Reports) ---")
    print(f"Extraction Range: 2026-03-01 to 2026-03-31 (Batched by 3 days)")
    
    try:
        for prop in properties:
            p_id = prop["id"]
            p_name = prop["name"]
            print(f"\n>>>> PROPERTY: {p_name} ({p_id})")
            
            for config in report_types:
                tag = config["tag"]
                first_metric = config["metrics"][0]
                print(f"  Report: {tag}")
                
                all_batches_data = []

                for start, end in march_batches:
                    print(f"    Fetching batch: {start} to {end}...", end=" ", flush=True)
                    
                    # Initialize using GA4_CoreEngine
                    builder = BuildReport(
                        property_id=p_id,
                        ga_dimensions=config["dimensions"],
                        ga_metrics=config["metrics"],
                        start_date=start,
                        end_date=end,
                        creds_path=CREDS_PATH
                    )
                    
                    # Sort Descending by first metric
                    sort_order = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name=first_metric), desc=True)]
                    
                    # FETCH ALL DATA (up to 250k rows per batch)
                    try:
                        df_batch = builder.run_report(limit=250000, order_bys=sort_order)
                        all_batches_data.append(df_batch)
                        print(f"Done ({len(df_batch)} rows)")
                        
                        # Wait between batches (requested by user)
                        if (start, end) != march_batches[-1]: 
                            print(f"    Waiting 10 seconds before next batch...")
                            time.sleep(10)
                    except Exception as batch_error:
                        print(f"FAILED: {batch_error}")
                
                # Combine batches
                if all_batches_data:
                    final_df = pl.concat(all_batches_data, how='vertical', rechunk=True)
                    
                    # Final export name: march_2026_[PROPERTY]_[TYPE]_full.csv
                    output_file = f"march_2026_{p_name.lower()}_{tag}_full.csv"
                    output_path = os.path.join(BASE_DIR, output_file)
                    
                    final_df.write_csv(output_path)
                    print(f"    >>> Total Exported: {len(final_df)} rows to: {output_file}")
                else:
                    print(f"    >>> No data found for {p_name} - {tag}")

        print("\n--- ALL MARCH DATA EXPORTED SUCCESSFULLY ---")

    except Exception as e:
        print(f"\nExecution Error: {e}")

if __name__ == '__main__':
    main()
