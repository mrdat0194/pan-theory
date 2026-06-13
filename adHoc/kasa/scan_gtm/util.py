from bs4 import BeautifulSoup
import urllib3
import re
import polars as pl
http = urllib3.PoolManager()

lambda _: _()
def get_content_html(sblink:str,gcode:str) -> str:
    try:
        response = http.request('GET', sblink)
        soup = BeautifulSoup(response.data,"html.parser")
        text = soup.find_all(name='script')
        #
        full_text = str()
        for i in range (0,len(text)):
            try: full_text+= text[i].get_text()
            except: None
        code_conf = re.findall(f"{gcode}'".strip(), full_text.strip())
        if code_conf == []: return 'No'
    except: return 'No'
    return 'Yes'

def convertTuple(tup:str) -> str:
    str = ''.join(tup)
    return str

df_linkes = pl.read_csv("tag-coverage-GTM-NQXT76Z.csv",columns="URL")

list_verification = []
for t in range(1,11):
    for i in range(0+((t-1)*1000),1000+((t-1)*1000)): list_verification.append(get_content_html(convertTuple(df_linkes.rows()[i]),"GTM-NQXT76Z"))
    df_verification=pl.DataFrame(list_verification)
    df_verification.to_pandas().to_csv("df_verification_b"+str(t)+".csv")
    list_verification = []


