import pandas as pd 
import numpy as np 
#import cPickle as pickle
import pickle
import os 
import csv
from tqdm import tqdm

def get_selected_words(x_single, score, id_to_word, k): 
    selected_words = {} # {location: word_id}

    selected = np.argsort(score)[-k:] 
    selected_k_hot = np.zeros(400)
    selected_k_hot[selected] = 1.0

    x_selected = (x_single * selected_k_hot).astype(int)
    return x_selected 

def create_dataset_from_score(x, scores, k):
    with open('data/id_to_word.pkl','rb') as f:
        id_to_word = pickle.load(f)
    new_data = []
    new_texts = []
    for i, x_single in enumerate(x):
        x_selected = get_selected_words(x_single, 
            scores[i], id_to_word, k)

        new_data.append(x_selected) 

    np.save('data/x_val-L2X.npy', np.array(new_data))

def calculate_acc(pred, y):
    return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))

def get_selected_words_group(x_single, group_selection, id_to_word, filtered=True):
    selected_words = []
    indices = np.where(group_selection > 0)[0].tolist()
    #print(indices)
    #print(x_single.shape)
    #print(group_selection.shape)
    for idx in indices:
        #print(idx)

        word_id = x_single[idx]
        if filtered:
            if word_id in range(3):
                continue
        #print(word_id)
        word = id_to_word[word_id]
        selected_words.append(word)
    return selected_words


def create_dataset_from_group_score(x, scores, filtered):
    with open('data/id_to_word.pkl','rb') as f:
        id_to_word = pickle.load(f)
    num_exp, num_groups, num_features = scores.shape
    explain_list = []
    for i, x_single in tqdm(enumerate(x)):
        x_dict = {}
        for grp in range(num_groups):
            grp_score = scores[i, grp, :]
            selected_words = get_selected_words_group(x_single, grp_score, id_to_word, filtered=filtered)
            x_dict[grp] = selected_words
        explain_list.append(x_dict)
    return explain_list
