import os
import pdb
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten
from keras.layers import  Dropout, GRU, Bidirectional, TimeDistributed, LSTM
from keras.models import Model
from keras import regularizers
from keras.utils.np_utils import to_categorical
from .defination import AttLayer
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import  Embedding
from gensim.models import Word2Vec, KeyedVectors
### self defined functions
from src.functions import rmse
from src.data_load import glove_embedding_layer_gene,test_data_load
from src.data_load import regression_data_load as load
from gensim.models import Word2Vec
from nltk import tokenize



def word2vec_embedding_single(texts, embedding_model,trans_item):
    all_embeddings = []
    for text in texts:
        text_embedding = np.zeros ([trans_item.word_num_per_trans, trans_item.embedding_size])
        wordTokens = text_to_word_sequence(text)
        # for each document
        if len(wordTokens) > trans_item.word_num_per_trans:
            wordTokens = wordTokens[:trans_item.word_num_per_trans]
        sent_embedding =  np.random.random((trans_item.word_num_per_trans, trans_item.embedding_size))    
        for word, i in enumerate(wordTokens):
            if word in embedding_model:
                sent_embedding[i] = embedding_model[word]
        if len(wordTokens)<trans_item.word_num_per_trans:  
            sent_embedding = np.concatenate ([sent_embedding, np.zeros([trans_item.word_num_per_trans-sent_embedding.shape[0] ,trans_item.embedding_size])])
        all_embeddings.append(text_embedding)
    all_embeddings = np.asarray(all_embeddings)
    return all_embeddings



def word2vec_embedding_hie(texts, embedding_model,trans_item):
    all_embeddings = []
    for text in texts:
        text_embedding = np.zeros ([trans_item.sent_nums, trans_item.word_num_per_sent, trans_item.embedding_size])
        sentences = tokenize.sent_tokenize(text)
        # for each document
        for j, sent in enumerate(sentences):
            if j < trans_item.sent_nums:
                # words in each sentence
                wordTokens = text_to_word_sequence(sent)
                if len(wordTokens) == 0:
                    continue
                if len(wordTokens) > trans_item.word_num_per_sent:
                    wordTokens = wordTokens[:trans_item.word_num_per_sent]
                sent_embedding =  np.random.random((trans_item.word_num_per_sent, trans_item.embedding_size))
                for word,i in enumerate(wordTokens):
                    if word in embedding_model:
                        sent_embedding[i] = embedding_model[word]
                if len(wordTokens)<trans_item.word_num_per_sent:  
                    sent_embedding = np.concatenate ([sent_embedding, np.zeros([trans_item.word_num_per_sent-sent_embedding.shape[0] ,trans_item.embedding_size])])
                text_embedding[j]= sent_embedding
        all_embeddings.append(text_embedding)
    all_embeddings = np.asarray(all_embeddings)
    return all_embeddings

def word2vec_model(texts, trans_item):
    all_words = []
    for text in texts:
        sentences = tokenize.sent_tokenize(text)
        for j, sent in enumerate(sentences):
            if j < trans_item.sent_nums:
                wordTokens = text_to_word_sequence(sent)
            all_words.append(wordTokens)
    embedding_model = Word2Vec(all_words, min_count=1)
    embedding_model.save("output/word2vec.model")
    return embedding_model

def word2vec_BiRNN_Att(trans_item, model_item):
    sequence_input = Input(shape=(trans_item.word_num_per_trans, trans_item.embedding_size), dtype='float32')
    x = Bidirectional(LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(sequence_input)
    x = Dropout(model_item.drop_rate)(x)
    x = AttLayer(model_item.Att_DIM, name = 'attention')(x)
    preds = Dense(model_item.label_num, activation='softmax')(x)
    model = Model(sequence_input, preds)
    print(model.summary())
    return model

def word2vec_BiRNN(trans_item, model_item):
    sequence_input = Input(shape=(trans_item.word_num_per_trans, trans_item.embedding_size), dtype='float32')
    x = Bidirectional(LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(sequence_input)
    x = Dropout(model_item.drop_rate)(x)
    x = Dense(model_item.Dense_DIM, activation='relu')(x)
    x = Flatten()(x)
    preds = Dense(model_item.label_num, activation='softmax')(x)
    model = Model(sequence_input, preds)
    print(model.summary())
    return model

def word2vec_hie_Att(trans_item, model_item):
    ############word-level#############
    sentence_input = Input(shape=(trans_item.word_num_per_sent, trans_item.embedding_size), dtype='float32')
    x = Bidirectional(LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(sentence_input)
    x = Dropout(model_item.drop_rate)(x)
    x = Dense(model_item.Dense_DIM, activation='relu')(x)
    output = AttLayer(model_item.Att_DIM)(x)
    sentEncoder = Model(sentence_input, output)
    print(sentEncoder.summary())
    ############sentence-level: Bi-rnn############
    review_input = Input(shape=(trans_item.sent_nums, trans_item.word_num_per_sent, trans_item.embedding_size), dtype='float32')
    x = TimeDistributed(sentEncoder)(review_input)
    x = Bidirectional(LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(model_item.drop_rate)(x)
    x = AttLayer(model_item.Att_DIM,name = 'attention')(x)
    preds = Dense(model_item.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(model_item.l2))(x)
    model = Model(review_input, preds)
    print(model.summary())
    return model

def word2vec_hie(trans_item,model_item):
    ############word-level#############
    sentence_input = Input(shape=(trans_item.word_num_per_sent, trans_item.embedding_size), dtype='float32')
    x = Bidirectional(LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(sentence_input)
    x = Dropout(model_item.drop_rate)(x)
    x = Dense(model_item.Dense_DIM, activation='relu')(x)
    x = Flatten()(x)
    sentEncoder = Model(sentence_input, output)
    print(sentEncoder.summary())
    ############sentence-level: Bi-rnn############
    review_input = Input(shape=(trans_item.sent_nums, trans_item.word_num_per_sent, trans_item.embedding_size), dtype='float32')
    x = TimeDistributed(sentEncoder)(review_input)
    x = Bidirectional(LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(model_item.drop_rate)(x)
    x = Flatten()(x)
    preds = Dense(model_item.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(model_item.l2))(x)
    model = Model(review_input, preds)
    print(model.summary())
    return model


def word2vec_regression_func(trans_item,model_item,mmse_dict, path_item):
    test_lists = open(path_item.test_list).readlines()
    test_labels, test_texts = load(test_lists, path_item.test_transcript_path,mmse_dict)
    
    dev_rmse = []
    test_rmse = []
    for idx in range(model_item.Fold):
        print('fold ' + str(idx))
        path_item.res_f.write('fold: ' + str(idx) + '\n')
        ### data_preparation
        train_lst_path = os.path.join(path_item.list_dir, str(idx), 'trans_train')
        train_lists = open(train_lst_path).readlines()
        dev_lst_path = os.path.join(path_item.list_dir, str(idx), 'trans_dev')
        dev_lists = open(dev_lst_path).readlines()


        train_labels, train_texts = load(train_lists, path_item.train_transcript_path,mmse_dict)
        dev_labels, dev_texts = load(dev_lists, path_item.train_transcript_path,mmse_dict)

        all_texts = train_texts + dev_texts + test_texts
        ## embedding generation     
        #embedding_model = KeyedVectors.load_word2vec_format(path_item.word2vec_path, binary=True)    
        embedding_model=word2vec_model(all_texts, trans_item)
        ## model initialization
        model_folder_ex = os.path.join(path_item.model_folder,model_item.model_type)
        if os.path.exists(model_folder_ex) == False:
            os.makedirs(model_folder_ex)
        model_saved_path = os.path.join(model_folder_ex, str(idx))
        
        
        if model_item.model_type == 'Hie' or model_item.model_type == 'Hie_Att':
            train_embedding = word2vec_embedding_hie(train_texts, embedding_model, trans_item)
            dev_embedding = word2vec_embedding_hie(dev_texts, embedding_model, trans_item)
            test_embedding = word2vec_embedding_hie(test_texts, embedding_model, trans_item)
            if model_item.model_type == 'Hie' :
                model = word2vec_hie(trans_item, model_item)
            else :
                model = word2vec_hie_Att(trans_item, model_item)
        else:
            train_embedding = word2vec_embedding_single(train_texts, embedding_model, trans_item)
            dev_embedding = word2vec_embedding_single(dev_texts, embedding_model, trans_item)
            test_embedding = word2vec_embedding_single(test_texts, embedding_model, trans_item) 
            if model_item.model_type == 'BiRNN' :
                model = word2vec_BiRNN(trans_item, model_item)
            else :
                model = word2vec_BiRNN_Att(trans_item, model_item)    
                 
        checkpoint = ModelCheckpoint(model_saved_path, monitor='val_rmse', verbose=2, save_best_only=True,
                                         mode='min', save_weights_only=True)                
        model.compile(loss='mse',
                          optimizer='adam',
                          metrics=[rmse])                
                
        model.fit(train_embedding, train_labels,
                  batch_size=model_item.batch_size,
                  epochs=model_item.epochs,
                  validation_data=(dev_embedding, dev_labels),
                  callbacks=[checkpoint],
                  shuffle=True)
        model.load_weights(model_saved_path)                                    
        
        dev_loss, dev_r = model.evaluate(dev_embedding, dev_labels, batch_size=model_item.batch_size)
        test_loss, test_r = model.evaluate(test_embedding, test_labels, batch_size=model_item.batch_size)
        dev_rmse.append(dev_r*30)
        test_rmse.append(test_r*30)
        
    dev_rmse = np.asarray(dev_rmse)
    dev_rmse_mean = np.mean(dev_rmse, axis=0)
    
    test_rmse = np.asarray(test_rmse)
    test_rmse_mean = np.mean(test_rmse, axis=0)

    print('dev_rmse:'+str(dev_rmse_mean)+'\n')
    print('test_rmse:'+str(test_rmse_mean)+'\n')