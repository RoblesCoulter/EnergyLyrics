import numpy as np
import pandas as pd
from English_to_IPA import conversion
import re
from num2words import num2words

eSPEDict = dict()
with open('eSPEPhonologicalTableV2') as openFile:
    for line in openFile.readlines():
        line = line.strip().split('\t')
        eSPEDict[line[0]] = np.array([eval(x) for x in line[1:]])

def clean_lyrics(lyrics):
    punct_str = '!"#$%&\'()*+,-./:;<=>/?@[\\]^_`{|}~«» '
    splitted_lyrics = re.sub(' +', ' ', lyrics.replace('\n','').replace('-', ' ')).split(' ')
    return [word.strip(punct_str).lower() for word in splitted_lyrics]

def is_number(word):
    try:
        int(word)
        return True
    except ValueError:
        pass
    return False

def change_number_to_string(number):
    return num2words(number)
    
def get_score_from_lyric(lyrics):
    word_count = 0
    score = np.zeros((8,), dtype=np.int)
    words = clean_lyrics(lyrics)
    for word in words:
        if is_number(word):
            word = change_number_to_string(int(word))
        if word == '':
            continue
        cmu, ipa = conversion.convert(word)
        ipa = re.sub('[ˌˈ ]' ,'' ,ipa)
        if '*' in ipa:
            continue
        word_count += 1
        i = 0
        while i < len(ipa):
            if i == len(ipa)-1:
                sym = ipa[i]
                score += eSPEDict[sym]
                i += 1
            else:
                try:
                    sym = ipa[i] + ipa[i+1]
                    eSPEDict[sym]
                    i += 2
                except KeyError:
                    sym = ipa[i]
                    eSPEDict[sym]
                    i += 1
    score = np.append(score, word_count)
    return score            

lyrics_data = pd.read_csv("LyricsFreak.csv")

energy_lyrics = pd.DataFrame(np.zeros(shape=(len(lyrics_data), 9)),columns=['A', 'B', 'C', 'D', 'E','F','G','H','word_count'])
lyric_id = 0

for lyric in lyrics_data["text"]:
    try:
        energy_lyrics.loc[lyric_id] = get_score_from_lyric(lyric)
        lyric_id = lyric_id + 1
        print(lyric_id)
    except IndexError:
        print(lyric)
        print(clean_lyrics(lyric))
        break

energy_lyrics.to_csv('EnergyScores.csv')

    
