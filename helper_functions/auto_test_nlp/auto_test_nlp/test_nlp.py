#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import requests
import pprint
import urllib.parse
import json

try:
    with open("automated_test.txt") as f_test:
        test_auto = f_test.readlines()
except FileNotFoundError:
    test_auto = []

f_intent = open('automated_fail_intent.txt','w')
f_tag = open('automated_fail_tag.txt','w')
f_gnj_place = open('automated_fail_gnj_place.txt','w')

count_intent=0
count_tag=0
count_gnj_place=0
total_processed=0

for i, sentence in enumerate(test_auto, 1):
    clean_sentence=sentence.split("\n")[0]
    url="http://testgnjnlp.herokuapp.com/api/chunks/vi/"+urllib.parse.quote_plus(clean_sentence.split(";")[0])
    url=url.replace("+","%20")

    try:
        response2 = requests.get(url, timeout=10)
        response2.raise_for_status()
        resp_json = response2.json()

        predict_intent=resp_json['entities']['intent'][0]['value']
        predict_tag=[]
        for tag in resp_json['entities']['tag']:
            predict_tag.append(tag['value'])
        predict_gnj=[]
        for gnj in resp_json['entities']['gnj_place']:
            predict_gnj.append(gnj['value'])
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f'Error processing case #{i}: {e}')
        continue

    total_processed += 1
    print(f'Testing case #{i}: {clean_sentence.split(";")[0]}')
    valid_intent=clean_sentence.split(";")[1]    
    if (clean_sentence.split(";")[2]=='null'):
        valid_tag=[]
    else:
        valid_tag=clean_sentence.split(";")[2].split(",")    
    if (clean_sentence.split(";")[3]=='null'):
        valid_gnj_place=[]
    else:
        valid_gnj_place=clean_sentence.split(";")[3].split(",")

    valid_tag.sort()
    predict_tag.sort()
    valid_gnj_place.sort()
    predict_gnj.sort()
    correct_tag=True
    correct_gnj=True
    correct_intent=True
    if (predict_intent!=valid_intent):
        correct_intent=False
    if (len(valid_tag)!=len(predict_tag)):
        correct_tag=False
    else:
        for j in range(0,len(valid_tag)):
            if(valid_tag[j]!=predict_tag[j]):
                correct_tag=False
                break
    if (len(valid_gnj_place)!=len(predict_gnj)):
        correct_gnj=False
    else:
        for j in range(0,len(valid_gnj_place)):
            if(valid_gnj_place[j]!=predict_gnj[j]):
                correct_gnj=False
                break
    if(correct_intent):
        count_intent +=1
    else:
        f_intent.write(clean_sentence.split(";")[0]+":"+valid_intent+"<===>"+predict_intent+"\n")
    if(correct_tag):
        count_tag +=1
    else:
        f_tag.write(clean_sentence.split(";")[0]+":"+','.join(valid_tag)+"<===>"+','.join(predict_tag)+"\n")
    if(correct_gnj):
        count_gnj_place +=1
    else:
        f_gnj_place.write(clean_sentence.split(";")[0]+":"+','.join(valid_gnj_place)+"<===>"+','.join(predict_gnj)+"\n")

print("Correct Intent: "+str(count_intent)+"/"+str(total_processed))
print("Correct Tag: "+str(count_tag)+"/"+str(total_processed))
print("Correct Gnj Place: "+str(count_gnj_place)+"/"+str(total_processed))

f_intent.close()
f_tag.close()
f_gnj_place.close()
    


#result["entities"]["gnj_place"].sort(key=lambda x: x['confidence'], reverse=True)
    
