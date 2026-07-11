#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import requests
import pprint
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

with open("automated_test.txt") as f_test:
    test_auto = f_test.readlines()

def process_sentence(args):
    sentence, count = args
    clean_sentence=sentence.split("\n")[0]
    url="http://testgnjnlp.herokuapp.com/api/chunks/vi/"+urllib.parse.quote_plus(clean_sentence.split(";")[0])
    url=url.replace("+","%20")

    try:
        response2  = requests.get(url)
        response_json = response2.json()
    except Exception as e:
        response_json = {'entities': {'intent': [{'value': ''}], 'tag': [], 'gnj_place': []}}

    print('Testing case #'+str(count)+": "+clean_sentence.split(";")[0])

    parts = clean_sentence.split(";")
    valid_intent = parts[1] if len(parts) > 1 else ""

    if len(parts) > 2 and parts[2] == 'null':
        valid_tag=[]
    elif len(parts) > 2:
        valid_tag=parts[2].split(",")
    else:
        valid_tag = []

    if len(parts) > 3 and parts[3] == 'null':
        valid_gnj_place=[]
    elif len(parts) > 3:
        valid_gnj_place=parts[3].split(",")
    else:
        valid_gnj_place = []

    try:
        predict_intent=response_json['entities']['intent'][0]['value']
    except (KeyError, IndexError):
        predict_intent=""

    predict_tag=[]
    try:
        for tag in response_json['entities'].get('tag', []):
            predict_tag.append(tag['value'])
    except KeyError:
        pass

    predict_gnj=[]
    try:
        for gnj in response_json['entities'].get('gnj_place', []):
            predict_gnj.append(gnj['value'])
    except KeyError:
        pass

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
        for i in range(0,len(valid_tag)):
            if(valid_tag[i]!=predict_tag[i]):
                correct_tag=False
                break

    if (len(valid_gnj_place)!=len(predict_gnj)):
        correct_gnj=False
    else:
        for i in range(0,len(valid_gnj_place)):
            if(valid_gnj_place[i]!=predict_gnj[i]):
                correct_gnj=False
                break

    return {
        'clean_sentence': clean_sentence,
        'valid_intent': valid_intent,
        'predict_intent': predict_intent,
        'valid_tag': valid_tag,
        'predict_tag': predict_tag,
        'valid_gnj_place': valid_gnj_place,
        'predict_gnj': predict_gnj,
        'correct_intent': correct_intent,
        'correct_tag': correct_tag,
        'correct_gnj': correct_gnj
    }

f_intent = open('automated_fail_intent.txt','w')
f_tag = open('automated_fail_tag.txt','w')
f_gnj_place = open('automated_fail_gnj_place.txt','w')

count_intent=0
count_tag=0
count_gnj_place=0

tasks = [(sentence, i + 1) for i, sentence in enumerate(test_auto)]

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_sentence, tasks))

for result in results:
    if(result['correct_intent']):
        count_intent +=1
    else:
        f_intent.write(result['clean_sentence'].split(";")[0]+":"+result['valid_intent']+"<===>"+result['predict_intent']+"\n")

    if(result['correct_tag']):
        count_tag +=1
    else:
        f_tag.write(result['clean_sentence'].split(";")[0]+":"+','.join(result['valid_tag'])+"<===>"+','.join(result['predict_tag'])+"\n")

    if(result['correct_gnj']):
        count_gnj_place +=1
    else:
        f_gnj_place.write(result['clean_sentence'].split(";")[0]+":"+','.join(result['valid_gnj_place'])+"<===>"+','.join(result['predict_gnj'])+"\n")

f_intent.close()
f_tag.close()
f_gnj_place.close()

print("Correct Intent: "+str(count_intent)+"/"+str(len(test_auto)))
print("Correct Tag: "+str(count_tag)+"/"+str(len(test_auto)))
print("Correct Gnj Place: "+str(count_gnj_place)+"/"+str(len(test_auto)))
