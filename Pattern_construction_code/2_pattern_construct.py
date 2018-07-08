import os
import re
import sys
import math
import time
import pickle
import pandas as pd
from nltk import ngrams
from collections import defaultdict, Counter
from nltk.tokenize import TweetTokenizer

import multiprocessing as mp

sys.setrecursionlimit(100000)

 # turing point from the figure drawn by excel

emotion_num = 4

cc_threshold = 0.2 # 0.81 # 591 nodes

ec_threshold = 0.67 #0.02 # 0.0022 # 720 nodes

# file_path = './datasets/elvis_after_preprocessing_dataset'

file_path = '../data/train_4emo.tsv'

target_dir = './luis_network/'

out_dir = './luis_pattern_half/'

out_path = out_dir + 'patterns_ignore_5'

tknzr = TweetTokenizer()


def replace_Token(tweet):
    tweet = re.sub('((https?:\/\/)|(pic.twitter))\S+','URLTOK',tweet.lower().strip()) # url
    tweet = re.sub('@(?:[a-zA-Z0-9_]+)', '<M>', tweet) # mention
    tweet = re.sub('#(?:[a-zA-Z0-9_]+)', '<H>', tweet) # hashtag
    tweet = tweet.replace('\n'," ")
    return tweet


def get_pattern_scores(tweet, emotion, tknzr, cw_list, tw_list,):
    """
    RETURN

    [LIST] of pattern scores tuple, the length of LIST depend on the pattern situation

    [(pattern, emotion, tw, pf_value, ef_value, div_value), ... ]

    """
    try:
        words = tknzr.tokenize(replace_Token(tweet))
    except:
        print(tweet)
        try:
            words = tweet.split(' ')
        except:
            words = []
    labeled = [[w, label_words(w, cw_list, tw_list)] for w in words] 
    
    words_ngram = list(ngrams(labeled, 3)) 
    

    return_list = [] # multiple result


    for gram in words_ngram:

        label_list = [gram[0][1], gram[1][1], gram[2][1]] 
        label_count = Counter(label_list) 

         
        if 0 not in label_count and label_count[1] < 3 and label_count[2] < 2: # no 3
            if label_count[1] == 2 and label_count[2] == 1: # [1, 2, 1] or [1, 1, 2] or [2, 1, 1] 
                pattern, tw = transfer_tw(gram) 
                
                # pattern_frequency[pattern][emotion] += 1 
                # pattern_ef[pattern][emotion] =1
                # diversity[pattern][tw] += 1 

                return_list.append((pattern, emotion, tw, 1, 1, 1))

                 
            if label_count[3] == 1: # one label 3 
                idx = label_list.index(3) 
                if label_count[1] == 2: # e.g.[1, 1, 3] -> [1, 1, 2] 
                    gram[idx][1] = 2 
                     
                if label_count[1] == 1: # e.g.[1, 2, 3] -> [1, 2, 1] 
                    gram[idx][1] = 1 
                     
                pattern, tw = transfer_tw(gram) 
                
                # pattern_frequency[pattern][emotion] += 1 
                # pattern_ef[pattern][emotion] =1
                # diversity[pattern][tw] += 1 

                return_list.append((pattern, emotion, tw, 1, 1, 1))
                 
            if label_count[3] == 2: # two label 3 
                
                idx = [i for i, label in enumerate(label_list) if label == 3] 
                if label_count[2] == 1: # e.g.[2, 3, 3] -> [2, 1, 1] 
                    for i in idx: 
                        gram[i][1] = 1 
                     
                    pattern, tw = transfer_tw(gram) 
                    
                    # pattern_frequency[pattern][emotion] += 1 
                    # pattern_ef[pattern][emotion] =1
                    # diversity[pattern][tw] += 1 

                    return_list.append((pattern, emotion, tw, 1, 1, 1))
                     
                if label_count[1] == 1: # e.g.[1, 3, 3] -> [1, 1, 2] or [1, 2, 1] 
                    
                    return_list = [] # multiple result
                    
                    # first situation 
                    gram[idx[0]][1] = 1 
                    gram[idx[1]][1] = 2 
                    pattern, tw = transfer_tw(gram) 
                    
                    # pattern_frequency[pattern][emotion] += 0.5 
                    # pattern_ef[pattern][emotion] =1
                    # diversity[pattern][tw] += 1 
                    return_list.append((pattern, emotion, tw, 0.5, 1, 1)) 

                    # second situation 
                    gram[idx[0]][1] = 2 
                    gram[idx[1]][1] = 1 
                    pattern, tw = transfer_tw(gram) 
                    
                    # pattern_frequency[pattern][emotion] += 0.5 
                    # pattern_ef[pattern][emotion] =1
                    # diversity[pattern][tw] += 1 

                    return_list.append((pattern, emotion, tw, 0.5, 1, 1))

                    
                     
            if label_count[3] == 3: # three label 3
                # e.g.[3, 3, 3] -> [1, 2, 1] or [1, 1, 2] or [2, 1, 1] 
                full_sets = [[[gram[0][0], 1], [gram[1][0], 2], [gram[2][0], 1]],  
                            [[gram[0][0], 1], [gram[1][0], 1], [gram[2][0], 2]], 
                            [[gram[0][0], 2], [gram[1][0], 1], [gram[2][0], 1]]]


                for s in full_sets: 
                    pattern, tw = transfer_tw(s) 
                    
                    # pattern_frequency[pattern][emotion] += 0.33 
                    # pattern_ef[pattern][emotion] =1
                    # diversity[pattern][tw] += 1 

                    return_list.append((pattern, emotion, tw, 0.33, 1, 1))

    return return_list

def get_pattern(df, cw_list, tw_list, multiprocess = True, chunk_size = 10000, normalize=False): 
    """

    RETURN
    1.
    pattern_frequency dictionary:{
        pattern: pattern_probability_of_all_the_pattern
    }

    2.
    diversity dictionary:{
        pattern:{
            wildcard_word : wildcard_word_count
        }
    }
    """
    pattern_frequency = defaultdict(lambda: defaultdict(lambda: 0))
    pattern_ef = defaultdict(lambda: defaultdict(lambda: 0))
    diversity = defaultdict(lambda: defaultdict(lambda: 0))

    total_freq = 0
    
    cores = mp.cpu_count()

    # pool = mp.Pool(processes = cores)
    if multiprocess:
        print('\nMatch pattern, using cores:', str(cores))
    else:
        print('\nMatch pattern, using single core')

    for i in range(0, df.shape[0], chunk_size):

        

        print("\nStart from {} \n".format(i))
        if i + chunk_size > df.shape[0]:
            chunk = df[i:]
        else:
            chunk = df[i:i+chunk_size]

        if multiprocess:
            pool = mp.Pool(processes = cores)
            
            results = [pool.apply_async(get_pattern_scores, args=(tweet, emotion, tknzr,cw_list, tw_list, )) for _,(emotion, tweet) in chunk.iterrows()] # for each three-gram

            pool.close()
            pool.join()
        else:
            results=[get_pattern_scores(tweet, emotion, tknzr,cw_list, tw_list) for _,(emotion, tweet) in chunk.iterrows()]
        
        print('\nReduce matching result')
        for j,res in enumerate(results):
            if multiprocess:
                pattern_scores_list = res.get()
            else:
                pattern_scores_list = res

            if len(pattern_scores_list) == 0: 
                continue

            # print('\nUpdate to dictionary\n')
            try:
                for pattern, emotion, tw, pf_value, ef_value, div_value in pattern_scores_list:
                
                    total_freq += pf_value
                    pattern_frequency[pattern][emotion] += pf_value
                    pattern_ef[pattern][emotion] = ef_value
                    diversity[pattern][tw] += div_value 
            except:
                print(j)

    if normalize:
        for p, d in pattern_frequency.items(): 
            for e, pf in d.items():
                pattern_frequency[p][e] = pf/total_freq 
    
    return pattern_frequency, diversity, pattern_ef

def label_words(word, cw_list, tw_list):
    if word in cw_list and word in tw_list:
        return 3 # Both
    elif word in tw_list:
        return 2 # PW
    elif word in cw_list:
        return 1 # CW
    else:
        return 0

def transfer_tw(gram):
    tw = None
    pattern = []
    for idx, item in enumerate(gram):
        if item[1] == 2:
            pattern.append('*')
            tw = item[0]
            continue
        
        pattern.append(item[0])
    return tuple(pattern), tw

def default_to_regular(d): 
    if isinstance(d, defaultdict): 
        d = {k: default_to_regular(v) for k, v in d.items()} 
    return d

def main():

    if 'pw.pkl' not in os.listdir(target_dir):

        print('\nRead gephi csv file\n')
        gephi_table = pd.read_csv(target_dir+'gephi_table.csv')
        print(gephi_table.shape)
        print(gephi_table.columns)


        pw_list = gephi_table[(gephi_table.clustering >= cc_threshold)].Label.values #& (gephi_table.triangles > 1)].Label.values
        cw_list = gephi_table[gephi_table.eigencentrality >= ec_threshold].Label.values

        print('CW count: {}\nPW count: {}'.format(len(cw_list), len(pw_list)))

        print('\nDump to pickle in ' + target_dir + '\n')

        with open(target_dir+'pw.pkl','wb') as f:
            pickle.dump(pw_list, f)
        
        with open(target_dir+'cw.pkl','wb') as f:
            pickle.dump(cw_list, f)

    else:
        print('\nLoad CW, PW from pickle\n')

        with open(target_dir + 'pw.pkl','rb') as f:
            pw_list = pickle.load(f)
        
        with open(target_dir +'cw.pkl','rb') as f:
            cw_list = pickle.load(f)


        print('CW count: {}\nPW count: {}'.format(len(cw_list), len(pw_list)))

    print('\nCW')
    print(cw_list[:5])


    print('\nPW')
    print(pw_list[:5])


    print('\nRead Documents\n')
    emotion_tweets = pd.read_csv(file_path, sep='\t',index_col=0)
    print(emotion_tweets.head(5))

    print('\nTweets: ', emotion_tweets.shape)

    process_tweets = emotion_tweets.iloc[:int(emotion_tweets.shape[0]/2)]

    print('\nProcessing: ', process_tweets.shape)

    print('\nStart to Calculate Pattern\n')

    start_time = time.time()

    pattern_frequency, diversity, pattern_ef = get_pattern(process_tweets, cw_list, pw_list)

    print('Total process time: ', str(time.time()-start_time))

    with open(out_dir + 'pattern_frequency.pkl','wb') as f:
            pickle.dump(default_to_regular(pattern_frequency), f)
        
    with open( out_path + 'diversity.pkl','wb') as f:
        pickle.dump(default_to_regular(diversity), f)

    with open(out_path + 'pattern_ef.pkl','wb') as f:
        pickle.dump(default_to_regular(pattern_ef), f)


    print('\nPattern Dumped!\n')



    print('\nCalculate pattern scores\n')

    with open(out_path,'w') as f:

        for pattern, d in pattern_frequency.items():
            if sum(d.values()) < 5: continue
            for emotion, pf in d.items():
                ief = emotion_num/len(pattern_ef.get(pattern))
                div = float(len(diversity.get(pattern)))

                pattern_score = math.log(pf+1.0, 10) * ief * math.log(div+1, 10)

                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(" ".join(pattern), emotion, pattern_score, float(pf), ief, div))

    print('\nFinish! output to ', out_path)




if __name__ == '__main__':
    main()
