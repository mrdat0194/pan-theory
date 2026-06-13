from datetime import datetime, timedelta

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

# Verify March 2026
batches = get_date_batches("2026-03-01", "2026-03-31", batch_days=3)
print(f"Total Batches: {len(batches)}")
for b in batches:
    print(b)
