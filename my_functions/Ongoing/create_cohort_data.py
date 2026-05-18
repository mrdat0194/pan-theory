import os
import pandas as pd
import glob
from operator import attrgetter

# Path to the data directory
data_dir = r"c:\Users\mrdat\PycharmProjects\pan-theory\main_def\data\Gonj"
files = glob.glob(os.path.join(data_dir, "Gonjoy*.txt"))

print(f"Found {len(files)} files.")

li = []
for filename in files:
    print(f"Reading {filename}...")
    try:
        df = pd.read_csv(filename)
        if 'from_id' in df.columns and 'time' in df.columns:
            li.append(df[['from_id', 'time']])
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if not li:
    print("No valid data files found.")
    exit()

print("Concatenating data...")
df_all = pd.concat(li, axis=0, ignore_index=True)
df_all = df_all.dropna(subset=['from_id', 'time'])

print("Processing dates...")
df_all['time'] = pd.to_datetime(df_all['time'])

# Get Cohort Month (first activity month for each user)
print("Calculating cohorts...")
df_all['CohortMonth'] = df_all.groupby('from_id')['time'].transform('min').dt.to_period('M')

# Get Activity Month
df_all['ActivityMonth'] = df_all['time'].dt.to_period('M')

# Calculate Period Index (months since join)
print("Calculating period indices...")
df_all['PeriodIndex'] = (df_all['ActivityMonth'] - df_all['CohortMonth']).apply(attrgetter('n'))

# Group by CohortMonth and PeriodIndex to get Freq
print("Grouping data...")
cohort_counts = df_all.groupby(['CohortMonth', 'PeriodIndex'])['from_id'].nunique().reset_index()
cohort_counts.columns = ['Var1', 'Var2', 'Freq']

# Sort for convenience
cohort_counts = cohort_counts.sort_values(['Var1', 'Var2'])

# Output to CSV
output_path = r"c:\Users\mrdat\PycharmProjects\pan-theory\main_def\data\biz\Table_Cohort_2.csv"
print(f"Saving to {output_path}...")
cohort_counts.to_csv(output_path)

print(f"File created successfully!")
print(cohort_counts.head())
