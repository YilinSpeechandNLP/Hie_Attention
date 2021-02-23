import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec
from nltk import tokenize
from keras.layers import Dense, Input, Flatten
from keras.layers import  Dropout, GRU, Bidirectional, TimeDistributed, LSTM
from keras.models import Model
from keras import regularizers
from keras.utils.np_utils import to_categorical
from .defination import AttLayer
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import  Embedding
#from keras_bert import Tokenizer
#import keras
#from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs

### self defined functions
from src.functions import f1,rc,pr
from src.data_load import test_data_load
from src.data_load import classification_data_load as load
import pdb
from gensim.models import Word2Vec, KeyedVectors


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


def word2vec_classification_func(trans_item,model_item,path_item):
    test_accuracy = []
    test_f_score = []
    
    val_accuracy = []
    val_f_score = []
    
    test_lists = open(path_item.test_list).readlines()
    test_labels, test_texts = load(test_lists, path_item.test_transcript_path)
    class_num = len(set(test_labels))
    test_labels = to_categorical(np.asarray(test_labels), class_num)
    for idx in range(model_item.Fold):
        print('fold ' + str(idx))
        path_item.res_f.write('fold: ' + str(idx) + '\n')
        ### list prepare
        train_lst_path = os.path.join(path_item.list_dir, str(idx), 'trans_train')
        train_lists = open(train_lst_path).readlines()
        dev_lst_path = os.path.join(path_item.list_dir, str(idx), 'trans_dev')
        dev_lists = open(dev_lst_path).readlines()
        # data load and prepare
        train_labels, train_texts = load(train_lists, path_item.train_transcript_path)
        dev_labels, dev_texts = load(dev_lists, path_item.train_transcript_path)
         # calculate the number of class in the data
        
        train_labels = to_categorical(np.asarray(train_labels), class_num)
        dev_labels = to_categorical(np.asarray(dev_labels), class_num)
        all_texts = train_texts+dev_texts+test_texts
        ## embedding generation
        #embedding_model=word2vec_model(all_texts, trans_item)
        embedding_model = KeyedVectors.load_word2vec_format(path_item.word2vec_path, binary=True)  

        
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
            else:
                model = word2vec_BiRNN_Att(trans_item, model_item)       

        checkpoint = ModelCheckpoint(model_saved_path, monitor='val_f1', verbose=2, save_best_only=True,
                                         mode='max', save_weights_only=True)
        model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[pr, rc, f1, 'accuracy'])

        model.fit(train_embedding, train_labels,
                  batch_size=model_item.batch_size,
                  epochs=model_item.epochs,
                  validation_data=(dev_embedding, dev_labels),
                  callbacks=[checkpoint],
                  shuffle=True)
        model.load_weights(model_saved_path)

        test_loss, test_pre, test_rec, test_f_m, test_acc = model.evaluate(test_embedding, test_labels, batch_size=model_item.batch_size)
        val_loss, val_pre, val_rec, val_f_m, val_acc = model.evaluate(dev_embedding, dev_labels, batch_size=model_item.batch_size)
        
        path_item.res_f.write(
                "eva test acc:" + str(round(test_acc, 4)) + " eva val F-score:" + str(round(test_f_m, 4)) + '\n\n')
        test_accuracy.append(test_acc)
        test_f_score.append(test_f_m)
        print('test_acc:'+str(test_acc)+'\n')
        print('test_f-score:'+str(test_f_m)+'\n')
        
        val_accuracy.append(val_acc)
        val_f_score.append(val_f_m)

        
    test_accuracy = np.asarray(test_accuracy)
    test_accuracy = np.mean(test_accuracy, axis=0)
    test_f_score = np.asarray(test_f_score)
    test_f_score = np.mean(test_f_score, axis=0)

    val_accuracy = np.asarray(val_accuracy)
    val_accuracy = np.mean(val_accuracy, axis=0)
    val_f_score = np.asarray(val_f_score)
    val_f_score = np.mean(val_f_score, axis=0)


    print('test_accuracy:'+str(test_accuracy)+'\n')
    print('test_f_score:'+str(test_f_score)+'\n')
    
    print('val_accuracy:'+str(val_accuracy)+'\n')
    print('val_f_score:'+str(val_f_score)+'\n')