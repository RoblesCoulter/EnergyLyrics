import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from six.moves import cPickle as pickle
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import itertools
no_alignment_file = [4764]
wrong_alignment = [3730]
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras_metrics

def extract_patterns(data,extract=False):
    if(extract):
        patterns = {}
        for index, row in data.iterrows():
            patterns[row['index']] = set(get_pattern([row['text']])[0].values())
            print('Extracted pattern from '+ row['index'] + ' index:'+ str(index))
            print('Size: ', len(patterns[row['index']]), 'Patterns size', len(patterns))
        try:
            print('Saving Pickle')
            with open('pickles/patterns/pattern.pickle','wb') as f:
                save = {
                    'patterns' : patterns
                }
                pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
                print('Successfully saved in pattern.pickle')
                return patterns
        except Exception as e:
            print('Unable to save data to pickle', e)
            print('Patterns probably not saved.')
            return patterns
    else:
        try:
            with open('pickles/patterns/pattern.pickle','rb') as f:
                save = pickle.load(f)
                patterns = save['patterns']
                del save
                returning = {}
                for key in list(data['index']):
                    returning[key] = patterns[key]
                return returning
        except Exception as e:
            print('Error loading base datasets pickle: ', e)
            
def clean_text(text, remove_actions= True):
    punct_str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~«»“…‘”'
    if(remove_actions):
        text = re.sub(r" ?\[[^)]+\]", "", text)
    for p in punct_str:
        text = text.replace(p,' ')
    text = re.sub(' +', ' ', text)
    return text.lower().strip()

def filter_word_count(data, n_count):
    return data[list(map(lambda x: len(x.split(' ')) >= n_count,data['text']))]


def remove_empty_patterns(data,patterns):
    empty_patterns = [k for k, v in patterns.items() if len(v) < 1]
    patterns = { k:v for k, v in patterns.items() if len(v) >= 1 }
    data = filter(lambda x: x[1]['index'] not in empty_patterns ,data.iterrows())
    data = pd.DataFrame.from_items(data).T
    return data,patterns

def remove_multiwildcard(patterns):
    for index, patt in patterns.items():
        flt_patt = {p for p in patt if p.split(' ').count('.+') == 1}
        patterns[index] = flt_patt
    return patterns

def load_data(word_count,emotional_mapping):
    # full = generate_IEMOCAP_df()
    data = pd.read_csv('data/IEMOCAP_sentences_votebased.csv',index_col=0)
    data['emotion_code'] = data['emotion'].map( emotional_mapping ).astype(int)
    # Take away fear, surprise,disgust, xxx and others. Not enough data
    data = data[data.emotion_code < 4]
    #Remove rows that don't have Alignment file
    try:
        data = data.drop(no_alignment_file)
    except Exception as e:
        print('Error at: ',e)
    # Remove rows that have wrong Alignment file
    try:
        data = data.drop(wrong_alignment)
    except Exception as e:
        print('Error at: ',e)
#     Clean Transcripts
    data['text'] = data['text'].apply(clean_text)
    # Filter Word Count
    data = filter_word_count(data, word_count)
    patterns = extract_patterns(data)
    data,patterns = remove_empty_patterns(data,patterns)
    patterns = remove_multiwildcard(patterns)
    return data,patterns

def load_acoustic_fullmatrices(extraction_type = 'full',extract_fd = False):
    if(extraction_type in ['full','wc','cw']):
        try:
            if(extract_fd):
                fullmfcc_matrix_fd = None
                fullrmse_matrix_fd = pd.read_pickle('pickles/patterns/'+extraction_type+'_rmse_matrix_fd.pickle')
                print('Successfully loaded '+extraction_type+' RMSE Matrix FULLDATA')
                fullzcr_matrix_fd = pd.read_pickle('pickles/patterns/'+extraction_type+'_zcr_matrix_fd.pickle')
                print('Successfully loaded '+extraction_type+' ZCR Matrix FULLDATA')   
                with open('pickles/patterns/'+extraction_type+'_mfcc20_matrix_fd.pickle','rb') as f:
                    save = pickle.load(f)
                    fullmfcc_matrix_fd = save['multimatrix']
                    del save
                print('Successfully loaded '+extraction_type+' MFCC Matrices FULLDATA')
                fullmfcc_matrix_fd.append(fullrmse_matrix_fd)
                fullmfcc_matrix_fd.append(fullzcr_matrix_fd)
                return fullmfcc_matrix_fd
            else:
                fullmfcc_matrix = None
                fullrmse_matrix = pd.read_pickle('pickles/patterns/'+extraction_type+'_rmse_matrix.pickle')
                print('Successfully loaded '+extraction_type+' RMSE Matrix')   
                fullzcr_matrix = pd.read_pickle('pickles/patterns/'+extraction_type+'_zcr_matrix.pickle')
                print('Successfully loaded '+extraction_type+' ZCR Matrix')
                with open('pickles/patterns/'+extraction_type+'_mfcc20_matrix.pickle','rb') as f:
                    save = pickle.load(f)
                    fullmfcc_matrix = save['multimatrix']
                    del save
                print('Successfully loaded '+extraction_type+' MFCC Matrices') 
                fullmfcc_matrix.append(fullrmse_matrix)
                fullmfcc_matrix.append(fullzcr_matrix)
                return fullmfcc_matrix
        except Exception as e:
            print('Error loading matrix: ', e)
    else:
        print('Error')
        return None,None

def get_frequency_vectors(data,patterns_list):
    patterns = extract_patterns(data)
    transcript_order = list(data['index'])
    frequency_vectors = []
    for index in patterns:
        frequency_vectors.append(np.isin(patterns_list,np.array(list(patterns[index]))))
    vectors = pd.DataFrame(frequency_vectors,columns=patterns_list,index=patterns.keys())
    vectors = vectors.loc[transcript_order]
    vectors = vectors * 1
    return vectors

seed = 7
np.random.seed(seed)
emotional_mapping = {'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3,'fru': 4,'exc': 5,'fea': 6,'sur': 7,'dis': 8, 'xxx':9,'oth':10}

data, patterns = load_data(3,emotional_mapping)
# x_train, x_test, y_train, y_test = train_test_split(data, data.emotion_code, test_size=TEST_SIZE)
try:
    with open('pickles/matrix_basedata.pickle','rb') as f:
        save = pickle.load(f)
        X_train = save['X_train']
        X_test = save['X_test']
        y_train = save['y_train']
        y_test = save['y_test']
        del save
except Exception as e:
    print('Error loading base datasets pickle: ', e)

y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

full_matrices = load_acoustic_fullmatrices(extraction_type='full',extract_fd = True)
wc_matrices = load_acoustic_fullmatrices(extraction_type='wc',extract_fd = True)
cw_matrices = load_acoustic_fullmatrices(extraction_type='cw',extract_fd = True)
########################################################################################
RMSE_INDEX = 20
ZCR_INDEX = 21
###########################################################################################

em_df = pd.read_pickle('pickles/patterns/pfief_matrix.pickle')

patterns_list = np.array(list(em_df.index))
print(len(em_df),len(full_matrices),len(wc_matrices),len(cw_matrices))

vectors = get_frequency_vectors(X_train,patterns_list)
test_vectors = get_frequency_vectors(X_test,patterns_list)

###########################################################################################
####### PARAMETERS ########
# EMBEDDING
EMBEDDING_DIM  = 4
MAX_SEQ_LENGTH = 170

# MODEL
FILTER_SIZES   = [1,1,1]
FEATURE_MAPS   = [150,150,150]
DROPOUT_RATE   = 0.2

# LEARNING
BATCH_SIZE     = 200
NB_EPOCHS      = 50
RUNS           = 1
VAL_SIZE       = 0.2
LEARNING_RATE  = 0.01

##############################################################################
# acoustic_matrix = full_matrices[RMSE_INDEX]
# acoustic_matrix = acoustic_matrix.fillna(np.max(acoustic_matrix))
NUM_CHANNELS = 22
acoustic_matrices = full_matrices[:20].copy()
acoustic_matrices.append(full_matrices[ZCR_INDEX])
for i,am in enumerate(acoustic_matrices):
    acoustic_matrices[i] = acoustic_matrices[i].fillna(np.max(acoustic_matrices[i]))

#######################################
full_data = []
for key, row in vectors.iterrows():
    final = []
    row_patt = [ i for i,v in row.iteritems() if v == 1]
    row_matrix = em_df.loc[row_patt,:].as_matrix()
    pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
    pad[:row_matrix.shape[0],:row_matrix.shape[1]] = row_matrix
    final.append(pad)
    ### ACU MATRICES ###
    for i,am in enumerate(acoustic_matrices):
        acu_matrix = am.loc[row_patt,:].as_matrix()
        acu_pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
        acu_pad[:acu_matrix.shape[0],:acu_matrix.shape[1]] = acu_matrix
        final.append(acu_pad)
    full_data.append(final)
    
test_full_data = []
for key, row in test_vectors.iterrows():
    final = []
    row_patt = [ i for i,v in row.iteritems() if v == 1]
    row_matrix = em_df.loc[row_patt,:].as_matrix()
    pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
    pad[:row_matrix.shape[0],:row_matrix.shape[1]] = row_matrix
    final.append(pad)
    ### ACU MATRICES ###
    for i,am in enumerate(acoustic_matrices):
        acu_matrix = am.loc[row_patt,:].as_matrix()
        acu_pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
        acu_pad[:acu_matrix.shape[0],:acu_matrix.shape[1]] = acu_matrix
        final.append(acu_pad)
    test_full_data.append(final)

acoustic_matrices = cw_matrices[:20].copy()
acoustic_matrices.append(cw_matrices[ZCR_INDEX])
for i,am in enumerate(acoustic_matrices):
    acoustic_matrices[i] = acoustic_matrices[i].fillna(np.max(acoustic_matrices[i]))

cw_data = []
for key, row in vectors.iterrows():
    final = []
    row_patt = [ i for i,v in row.iteritems() if v == 1]
    row_matrix = em_df.loc[row_patt,:].as_matrix()
    pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
    pad[:row_matrix.shape[0],:row_matrix.shape[1]] = row_matrix
    final.append(pad)
    ### ACU MATRICES ###
    for i,am in enumerate(acoustic_matrices):
        acu_matrix = am.loc[row_patt,:].as_matrix()
        acu_pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
        acu_pad[:acu_matrix.shape[0],:acu_matrix.shape[1]] = acu_matrix
        final.append(acu_pad)
    cw_data.append(final)
    
test_cw_data = []
for key, row in test_vectors.iterrows():
    final = []
    row_patt = [ i for i,v in row.iteritems() if v == 1]
    row_matrix = em_df.loc[row_patt,:].as_matrix()
    pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
    pad[:row_matrix.shape[0],:row_matrix.shape[1]] = row_matrix
    final.append(pad)
    ### ACU MATRICES ###
    for i,am in enumerate(acoustic_matrices):
        acu_matrix = am.loc[row_patt,:].as_matrix()
        acu_pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
        acu_pad[:acu_matrix.shape[0],:acu_matrix.shape[1]] = acu_matrix
        final.append(acu_pad)
    test_cw_data.append(final)


acoustic_matrices = wc_matrices[:20].copy()
acoustic_matrices.append(wc_matrices[ZCR_INDEX])

for i,am in enumerate(acoustic_matrices):
    acoustic_matrices[i] = acoustic_matrices[i].fillna(np.max(acoustic_matrices[i]))


wc_data = []
for key, row in vectors.iterrows():
    final = []
    row_patt = [ i for i,v in row.iteritems() if v == 1]
    row_matrix = em_df.loc[row_patt,:].as_matrix()
    pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
    pad[:row_matrix.shape[0],:row_matrix.shape[1]] = row_matrix
    final.append(pad)
    ### ACU MATRICES ###
    for i,am in enumerate(acoustic_matrices):
        acu_matrix = am.loc[row_patt,:].as_matrix()
        acu_pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
        acu_pad[:acu_matrix.shape[0],:acu_matrix.shape[1]] = acu_matrix
        final.append(acu_pad)
    wc_data.append(final)
    
test_wc_data = []
for key, row in test_vectors.iterrows():
    final = []
    row_patt = [ i for i,v in row.iteritems() if v == 1]
    row_matrix = em_df.loc[row_patt,:].as_matrix()
    pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
    pad[:row_matrix.shape[0],:row_matrix.shape[1]] = row_matrix
    final.append(pad)
    ### ACU MATRICES ###
    for i,am in enumerate(acoustic_matrices):
        acu_matrix = am.loc[row_patt,:].as_matrix()
        acu_pad = np.zeros((MAX_SEQ_LENGTH,EMBEDDING_DIM))
        acu_pad[:acu_matrix.shape[0],:acu_matrix.shape[1]] = acu_matrix
        final.append(acu_pad)
    test_wc_data.append(final)
import time
import multiap_cnn_model
# # BALANCED DATA
printing = {}
FILTER_SIZES_AR   = [[1,1,1]]
filter_sizes_names = ['1_1_1']
FEATURE_MAPS_AR   = [[150,150,150]]
feature_maps_names = ['150']
DROPOUT_RATE = 0.2
LEARNING_RATE  = 0.01
RUNS = 1 
DATA_AR = [ wc_data,cw_data]
TEST_DATA_AR = [test_wc_data,test_cw_data]
data_names = ['wc','cw']
MAX_SEQ_LENGTH = 170
for Findex,filterS in enumerate(FILTER_SIZES_AR):
    for Mindex, featureM in enumerate(FEATURE_MAPS_AR):
        for Dindex, dataV in enumerate(DATA_AR):
            FILTER_SIZES = filterS
            FEATURE_MAPS = featureM
            histories = []
            for i in range(RUNS):
                print('Running iteration %i/%i' % (i+1, RUNS))
                start_time = time.time()
                emb_layer = None

                model = multiap_cnn_model.build_cnn(
                    embedding_dim= EMBEDDING_DIM,
                    filter_sizes = FILTER_SIZES,
                    feature_maps = FEATURE_MAPS,  
                    max_seq_length = MAX_SEQ_LENGTH,
                    dropout_rate=DROPOUT_RATE,
                    num_channels=NUM_CHANNELS
                )

                model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adadelta(clipvalue=3,lr=LEARNING_RATE),
                    metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall()]
                )

                history = model.fit(
                    [dataV], y_train,
                    epochs=NB_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=([TEST_DATA_AR[Dindex]], y_test),
                    callbacks=[ModelCheckpoint('model-%i.h5'%(i+1), monitor='val_loss',
                                               verbose=0, save_best_only=True, mode='min'),
                               ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01)
                              ]
                )
                histories.append(history.history)
                print('Iteration', i+1)
                print("--- %s seconds on ---" % (time.time() - start_time))

            with open('history/mfcc20_zcr/_FS'+str(filter_sizes_names[Findex])+'_FM_'+str(feature_maps_names[Mindex])+'_data_'+str(data_names[Dindex])+'.pkl', 'wb') as f:
                pickle.dump(histories, f)

