import numpy as np
import tensorflow as tf 
import time 
import numpy as np 
import sys
from keras.datasets import imdb
from keras.preprocessing import sequence
import os
#import urllib2 
#import tarfile
#import zipfile 
#try:
#    import cPickle as pickle
#except:
import pickle
import os 
from utils import create_dataset_from_score, calculate_acc, get_selected_words_group, create_dataset_from_group_score

def load_data():
    """
    Load data if data have been created.
    Create data otherwise.

    """
    print('Loading data...')
    (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=5000, skip_top=0, index_from=3)
    word_to_id = imdb.get_word_index()
    word_to_id ={k:(v+3) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}

    print(len(x_train), 'train sequences')
    print(len(x_val), 'test sequences')

    print('Pad sequences (samples x time)')
    #x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    #x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
    y_train = np.eye(2)[y_train]
    y_val = np.eye(2)[y_val] 


    return x_train, y_train, x_val, y_val, id_to_word

def vec2str(vec, id_to_word):
    sentence = ""
    for word_id in vec:
        sentence += id_to_word[word_id] + " "
    return sentence

x_train, y_train, x_val, y_val, id_to_word = load_data()
print(vec2str(x_val[0], id_to_word))
print(vec2str(x_val[1], id_to_word))
