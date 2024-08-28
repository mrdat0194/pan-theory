import pandas as pd

file_names = ['df_verification_b1.csv', 'df_verification_b2.csv', 'df_verification_b3.csv', 'df_verification_b4.csv', 'df_verification_b5.csv', 'df_verification_b6.csv', 'df_verification_b7.csv', 'df_verification_b8.csv', 'df_verification_b9.csv', 'df_verification_b10.csv' ]

# Read each file and store in a list of DataFrames
df_full = pd.DataFrame()
for file in file_names:
    dfs = pd.read_csv(file)
    df_full = pd.concat([df_full,dfs], ignore_index=True)


# print(df_full)

df_full.to_csv('aggregate.csv', index=False)