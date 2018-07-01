from nltk import ngrams
from nltk import bigrams
from nltk.tokenize import TweetTokenizer
import nltk
import json
import codecs
import sys
import re
import pandas as pd

tknzr = TweetTokenizer()

def clean_text(text, remove_actions = True):
    punct_str = '!"#$%&()*+,-./:;<=>@[\\]^_`{|}~«»“…‘”'
    if(remove_actions):
        text = re.sub(r" ?\[[^)]+\]", "", text)
    for p in punct_str:
        text = text.replace(p,' ')
    text = re.sub(' +', ' ', text)
    return text.lower().strip()


def replace_Token(tweet):
    tweet = re.sub('((https?:\/\/)|(pic.twitter))\S+','URLTOK',tweet.lower().strip()) # url
    tweet = re.sub('@(?:[a-zA-Z0-9_]+)', '<M>', tweet) # mention
    tweet = re.sub('#(?:[a-zA-Z0-9_]+)', '<H>', tweet) # hashtag
    tweet = tweet.replace('\n'," ")
    return tweet

def weightedBigrams(path,mode):
    edges = {}
    with codecs.open(path,"r", "utf-8") as content_file:
        content = content_file.read()

    #mode 1 json
    #mode 2 plain
    if mode == 1: # json
        fullContent = []
        for line in content.split('\n'):
            try: 
                jsonContent = json.loads(line.strip())
                text = jsonContent.get('_source').get('tweet').get('text')
                sentence = replace_Token(text if text else '')
                fullContent.append(tknzr.tokenize(sentence))
            except:
                print(line)
            

    elif mode == 2: # plain text with tab sep
        fullContent = []
        for line in content.split('\n'):
            try:
                
                sentence = replace_Token(clean_text(line.split('\t')[1]))
                fullContent.append(tknzr.tokenize(sentence)) # only take the text part
            except:
                print(line)
    else: # plain text without tab sep
        fullContent = []
        for line in content.split('\n'):
            try:                
                sentence = replace_Token(line)
                fullContent.append(tknzr.tokenize(sentence))
            except:
                print(line)

    print(type(fullContent))
    print(len(fullContent))

    n_grams = []
    for line in fullContent:
        n_grams += list(ngrams(line, n=2))
#    for grams in n_grams:
#        print(grams)

    fdist = nltk.FreqDist(n_grams)

    print(len(fdist))
    out = codecs.open(path+"_edges","w", "utf-8-sig")
    maxi = 0.0
    foundMax = 0
    for k,v in fdist.most_common(int(len(fdist)*0.9)): #fdist.items():
        if v < 5: continue
        #print "'%s' - '%s' : %d"%(k[0],k[1],v)
        if len(k[0]) != 0 and len(k[1]) != 0 :
            # replace white space by a special mark
            if k[0] == " ":
                w1 = "_blank_"
            else:
                w1 = k[0]
            if k[1] == " ":
                w2 = "_blank_"
            else:
                w2 = k[1]
            if not foundMax:
                foundMax = 1
                maxi = float(v)
            #print "%s -> %s : %d - %f"%(k[0],k[1],v, float(v)/float(maxi))
            if(k[0] != "\n" and k[1] != "\n") :
                out.write("%s -> %s : %d - %f\n"%(k[0],k[1],v, float(v)/float(maxi)))
                edges["%s %s"%(w1,w2)]=float(v)/float(maxi)
    out.close
    return edges;

# weightedBigrams("./datasets/elvis_original_data.json", mode=1)
# weightedBigrams('./datasets/news', mode=3)
