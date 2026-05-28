import os
import polars as pl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
                
                # Prepare a function to fetch one batch so it can be mapped
                def fetch_batch(batch_idx, start, end):
                    print(f"    Fetching batch: {start} to {end}...", flush=True)
                    builder = BuildReport(
                        property_id=p_id,
                        ga_dimensions=config["dimensions"],
                        ga_metrics=config["metrics"],
                        start_date=start,
                        end_date=end,
                        creds_path=CREDS_PATH
                    )
                    sort_order = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name=first_metric), desc=True)]
                    try:
                        df_batch = builder.run_report(limit=250000, order_bys=sort_order)
                        print(f"    Done {start} to {end} ({len(df_batch)} rows)", flush=True)
                        return batch_idx, df_batch
                    except Exception as batch_error:
                        print(f"    FAILED {start} to {end}: {batch_error}", flush=True)
                        return batch_idx, None

                # Fetch all batches in parallel
                batch_results = [None] * len(march_batches)
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    for idx, (start, end) in enumerate(march_batches):
                        futures.append(executor.submit(fetch_batch, idx, start, end))

                    for future in as_completed(futures):
                        idx, df_batch = future.result()
                        if df_batch is not None:
                            batch_results[idx] = df_batch
                
                # Filter out failures
                all_batches_data = [df for df in batch_results if df is not None]

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
