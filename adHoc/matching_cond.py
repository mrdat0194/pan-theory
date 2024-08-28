import pandas as pd

df = pd.read_csv(r"D:\crawl_GTM\abc\abc.csv")
df["storage medium/ campaign"] = df[['storage_medium', 'storage_campaign']].agg('/ '.join, axis=1)
df = df.drop(columns=["storage_medium","storage_campaign"])
condition_dim = "HAN|SGN|DAD|VDO|HPH|VII|HUI|CXR|DLI|UIH|VCA|PQC|DIN|THD|VDH|VCL|TBB|PXU|BMV|VKG|CAH|VCS".replace("|",",").split(",")
match_checkpoint = list()
for i in range(0, len(df)):
    checkpoint =  [df['departure'].values[i].upper(),df['arrival'].values[i].upper()]
    match checkpoint:
        case checkpoint if checkpoint[0] in condition_dim and checkpoint[1] in condition_dim :
            match_checkpoint.append("Domestic")
        case _:
            match_checkpoint.append("International")

# df = df.drop(columns= ["departure","arrival"])
df  = pd.concat([df,pd.DataFrame(match_checkpoint,columns=["Trip"])],axis=1)

df.to_csv(r"D:\crawl_GTM\abc\abc_output.csv")
