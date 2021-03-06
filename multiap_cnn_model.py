# -*- coding: utf-8 -*-
"""
CNN model for text classification
This implementation is based on the original paper of Yoon Kim [1].
# References
- [1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
@author: Christopher Masch
"""

from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.models import Model

__version__ = '0.0.1'

def build_cnn(embedding_layer=None, num_words=None, num_channels= 1,
              embedding_dim=None, filter_sizes=[3,4,5],
              feature_maps=[100,100,100], max_seq_length=100, dropout_rate=None,num_emo = 4):
    """
    Building a CNN for text classification
    
    Arguments:
        embedding_layer : If not defined with pre-trained embeddings it will be created from scratch
        num_words       : Maximal amount of words in the vocabulary
        embedding_dim   : Dimension of word representation
        filter_sizes    : An array of filter sizes per channel
        feature_maps    : Defines the feature maps per channel
        max_seq_length  : Max length of sequence
        dropout_rate    : If defined, dropout will be added after embedding layer & concatenation
        
    Returns:
        Model           : Keras model instance
    """
    
    # Checks
    if len(filter_sizes)!=len(feature_maps):
        raise Exception('Please define `filter_sizes` and `feature_maps` with the same length.')
    # if not embedding_layer and (not num_words or not embedding_dim):
        # raise Exception('Please define `num_words` and `embedding_dim` if you not use a pre-trained embedding')
    print('Creating CNN %s' % __version__)
    print('#############################################')
    print('Embedding:    %s pre-trained embedding' % ('using' if embedding_layer else 'no'))
    # print('Vocabulary size: %s' % num_words)
    print('Embedding dim: %s' % embedding_dim)
    print('Filter sizes: %s' % filter_sizes)
    print('Feature maps: %s' % feature_maps)
    print('Max sequence: %i' % max_seq_length)
    print('Num Channels: %i' % num_channels)
    print('#############################################')  
    
    
    channels = []
    x_in = Input(shape=(num_channels,max_seq_length,embedding_dim), dtype='float64')
    if dropout_rate:
        # emb_layer  = Dropout(dropout_rate)(x_in)
        x = Dropout(dropout_rate)(x_in)
    #print('Added Dropout')    
    for ix in range(len(filter_sizes)):
        conv = create_channel(x, filter_sizes[ix], feature_maps[ix])
    #    print('Created Channel')
        channels.append(conv)
    
    # Concatenate all channels
    x = concatenate(channels)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = Activation('relu')(x)
    x = Dense(units=num_emo, activation='softmax')(x)
    return Model(inputs=x_in, outputs=x)
    
def create_channel(x, filter_size, feature_map):
    """
    Creates a layer working channel wise
    """
    x = Conv2D(filters=feature_map, kernel_size=(filter_size,filter_size),activation='relu', strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(1,1),padding='valid')(x)
    x = Flatten()(x)
    return x

