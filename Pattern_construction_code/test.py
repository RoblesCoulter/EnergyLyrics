import pandas as pd
import re 
from sklearn.model_selection import train_test_split


def clean_text(text, remove_actions = True):
    punct_str = '!"#$%&()*+,-./:;<=>@[\\]^_`{|}~«»“…‘”'
    if(remove_actions):
        text = re.sub(r" ?\[[^)]+\]", "", text)
    for p in punct_str:
        text = text.replace(p,' ')
    text = re.sub(' +', ' ', text)
    return text.lower().strip()

data = pd.read_csv('../data/IEMOCAP_sentences_votebased.csv',index_col=0)
emotional_mapping = {'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3,'fru': 4,'exc': 5,'fea': 6,'sur': 7,'dis': 8, 'xxx':9,'oth':10}
data['emotion_code'] = data['emotion'].map( emotional_mapping ).astype(int)
data = data[data.emotion_code < 4]
data['text'] = data['text'].apply(clean_text) # Removed actions
y = data.emotion_code
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
text_data = X_train[['emotion_code','text']]
text_data.to_csv('../data/train_4emo.tsv',sep='\t')
X_test.to_csv('../data/test_4emo.tsv', sep='\t')


# data = pd.read_csv('data/IEMOCAP_sentences.csv',index_col=0)
# data['emotion_code'] = data['emotion'].map( emotional_mapping ).astype(int)
# # Take away fear, surprise,disgust, xxx and others. Not enough data
# data = data[data.emotion_code < 4]
# #Remove rows that don't have Alignment file
# #     data = data.drop(no_alignment_file)
# # Clean Transcripts
# data['text'] = data['text'].apply(clean_text)
# # Filter Word Count
# data = filter_word_count(data, word_count)
# patterns = extract_patterns(data)
# data,patterns = remove_empty_patterns(data,patterns)
