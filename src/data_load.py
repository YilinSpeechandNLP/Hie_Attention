import numpy as np
import os
import re
from src.functions import clean_str
import pdb
from gensim.models import Word2Vec, KeyedVectors
import tensorflow as tf
from keras.layers import Embedding
from keras.engine.topology import Layer


    
def word2vec_embedding_layer_gene(word_index,  embedding_len, word2vec_path,embedding_size):
    #word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)  
    word2vec_model = Word2Vec.load(word2vec_path)
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_size))
    for word, j in list(word_index.items()):
        if word in word2vec_model:
            embedding_matrix[j]=word2vec_model.wv[word]
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_size,
                                weights=[embedding_matrix],
                                input_length=embedding_len,
                                trainable=True,
                                mask_zero=True)
    return embedding_layer

def embedding_idx_gene(glove_path):
    # embedding_index is a dictionary
    embeddings_index = {}
    coefs = None
    f = open(glove_path, 'rb')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # embedding_index is a dictionary
        embeddings_index[word] = coefs
    embedding_dim=len(coefs)
    return embeddings_index, embedding_dim

def glove_embedding_layer_gene(word_index, embedding_len, glove_path):
    embeddings_index, embedding_dim = embedding_idx_gene(glove_path)
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, j in list(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[j] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=embedding_len,
                                trainable=True,
                                mask_zero=True)
    return embedding_layer


def test_data_load(data_list, data_folder):
    texts = []
    for item in data_list:
        name = item.split(' ')[0]
        trans_path = os.path.join(data_folder, name)
        transcript = open(trans_path).readlines()[0]
        transcript = clean_str(transcript)
        texts.append(transcript)
    
    return texts

## classification data load
def classification_data_load(data_list, data_folder,label_dict):
    texts = []
    labels = []
    for item in data_list:
        name = item[:-1]
        label = label_dict[name]
        trans_path = os.path.join(data_folder, name)
        transcript = open(trans_path).readlines()[0]
        transcript = clean_str(transcript)
        texts.append(transcript)
        labels.append(label)
    
    return labels, texts


# regression data load
def regression_data_load(data_list, data_folder,mmse_dict):
    texts = []
    labels = []
    for item in data_list:
        name = item.split('\n')[0]
        label = mmse_dict.get(name) / 30
        trans_path = os.path.join(data_folder, name)
        transcript = open(trans_path).readlines()[0]
        transcript = clean_str(transcript)
        texts.append(transcript)
        labels.append(label)

    return labels, texts




