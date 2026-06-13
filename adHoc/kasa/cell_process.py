import pandas as pd
import os
import glob
from pathlib import Path
import pathlib
import shutil
import csv

#your folder path contain all files
path_landing = r"C:\Users\Kasatria\Documents\TASKS\cellphoneS\UPLOAD BIGQUERY\landing\0502"
path_campaign = r"C:\Users\Kasatria\Documents\TASKS\cellphoneS\UPLOAD BIGQUERY\campaign\0502"

landing_name = "landing_"
campaign_name = "campaign_"


def skiprows_landing(path):
    list_files = []
    files = Path(path).glob('*.csv')
    os.chdir(path)
    results = pd.DataFrame()
    #read files and skip 8 first rows
    for counter, current_file in enumerate (glob.glob("*.csv")):
        df = pd.read_csv(current_file, header=None, sep=",", skiprows = 8)
        results = pd.DataFrame(df)
    #export files to csv format

        print(f"----turn {counter} of landing----")
        results.to_csv(landing_name + str(counter) + ".csv", index=None, header=None, sep=",")

skiprows_landing(path_landing)



def appendFiles_landing(path):
    file_name = "landing_final.csv"
    os.chdir(path)
    files = Path(path).glob('*_*.csv')
    df = pd.DataFrame()
    for file in files:
    #     basename = os.path.basename(file)
    #     print(f"basename is {basename}")
    #     shutil.copy2(file, os.path.join(new_path, basename))
        df = pd.concat([df, pd.read_csv(file, header=None, sep=",")])
    #save csv file
    df.to_csv(file_name, index=None, header=None, sep=",")

appendFiles_landing(path_landing)
#appendFiles_landing(path_campaign)

def skiprows_campaign(path):
    list_files = []
    files = Path(path).glob('*.csv')
    os.chdir(path)
    results = pd.DataFrame()
    # read files and skip 8 first rows
    for counter, current_file in enumerate(glob.glob("*.csv")):
        df = pd.read_csv(current_file, header=None, sep=",", skiprows=8)
        results = pd.DataFrame(df)
        # export files to csv format

        print(f"----turn {counter} of campaign----")
        results.to_csv(campaign_name + str(counter) + ".csv", index=None, header=None, sep=",")

skiprows_campaign(path_campaign)