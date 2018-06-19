# This Python file uses the following encoding: utf-8
import os, sys

import numpy as np
import pandas as pd
from English_to_IPA import conversion
import re
import json
from matplotlib.font_manager import FontProperties
from num2words import num2words
valid_train = pd.read_csv('data/common_voice/cv-valid-train.csv',dtype={'ambiguous': str,'emotion':object},index_col=0)
valid_train.head()

import requests
import json

def get_deep_emotion(text):
    url = 'http://192.168.2.101:7878/api/get_emo'
    data = dict(input_tweets = text)
    resp = requests.post(url=url, data=data)
    r = json.loads(resp.text)
    return r

multiplier = 1000
iterations = len(valid_train)/multiplier

df = []
for i in range(iterations):
    init_i = i*multiplier
    last_i = (i+1)*multiplier
    if(i == iterations-1):
        last_i = len(valid_train)
    values = valid_train['text'][init_i:last_i]
    text = json.dumps(list(values))
    result = get_deep_emotion(text)
    temp_df = pd.DataFrame(data=result)
    df.append(temp_df)
    print('Added from',init_i,'to',last_i)
    concated_df = pd.concat(df)
    concated_df.to_csv('data/common_voice/deep-emo-train.csv')
    df = [concated_df]


concated_df = pd.concat(df)
concated_df.to_csv('data/common_voice/deep-emo-train.csv')
