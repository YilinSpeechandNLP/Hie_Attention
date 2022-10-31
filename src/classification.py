import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

import keras
#from keras_bert import Tokenizer
#import keras
#from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
import pdb
### self defined functions
from src.functions import f1,rc,pr
from src.data_load import glove_embedding_layer_gene, word2vec_embedding_layer_gene
from src.data_load import classification_data_load as load

def classification_func(trans_item,model_item,path_item):

    test_accuracy = []
    test_f_score = []
    test_percision = []
    test_recall = []
    val_accuracy = []
    val_f_score = []
    label_dict = np.load(path_item.label_dict_path, allow_pickle=True).item()
    for idx in range(model_item.Fold):
        np.random.seed(idx)
        print('fold ' + str(idx))
        path_item.res_f.write('fold: ' + str(idx) + '\n')
        ### list prepare
        train_lst_path = os.path.join(path_item.list_dir, str(idx), 'train_list')
        train_lists = open(train_lst_path).readlines()
        dev_lst_path = os.path.join(path_item.list_dir, str(idx), 'dev_list')
        dev_lists = open(dev_lst_path).readlines()
        test_lst_path = os.path.join(path_item.list_dir, str(idx), 'test_list')
        test_lists = open(test_lst_path).readlines()
        # data load and prepare
        
        train_labels, train_texts = load(train_lists, path_item.train_transcript_path, label_dict)
        dev_labels, dev_texts = load(dev_lists, path_item.train_transcript_path, label_dict)
        test_labels, test_texts = load(test_lists, path_item.train_transcript_path, label_dict)    
        

        train_labels = to_categorical(np.asarray(train_labels), model_item.class_num)
        dev_labels = to_categorical(np.asarray(dev_labels), model_item.class_num)
        test_labels = to_categorical(np.asarray(test_labels), model_item.class_num)
        all_texts = train_texts+dev_texts+test_texts
        
        ## tokenizer initialized
        tokenizer = Tokenizer(num_words=trans_item.max_NB_words)
        tokenizer.fit_on_texts(all_texts)
        # each word correspond to a int number 
        word_index = tokenizer.word_index

        # padding: regular the data format into the requirement matrix
        train_text_matrix = trans_item.matrix_gene(model_item.model_type,train_texts, tokenizer)
        dev_text_matrix =trans_item.matrix_gene(model_item.model_type,dev_texts, tokenizer)
        test_text_matrix = trans_item.matrix_gene(model_item.model_type,test_texts, tokenizer)

        
        if model_item.embedding_type == 'word2vec':
            embedding_layer = word2vec_embedding_layer_gene(word_index, trans_item.embedding_len,path_item.word2vec_path,trans_item.embedding_size)
        elif model_item.embedding_type == 'Glove':
            embedding_layer = glove_embedding_layer_gene(word_index, trans_item.embedding_len,path_item.glove_path)
        else:
            print('embedding type error')
            exit(1)
        ## model initialization
          
        model_folder_ex = os.path.join(path_item.model_folder,model_item.model_type)
        if os.path.exists(model_folder_ex) == False:
            os.makedirs(model_folder_ex)
   
        model_saved_path = os.path.join(model_folder_ex, str(idx))
        model = model_item.model_select(trans_item,embedding_layer)
        checkpoint = ModelCheckpoint(model_saved_path, monitor='val_f1', verbose=2, save_best_only=True,
                                         mode='max', save_weights_only=True)
        model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[pr, rc, f1, 'accuracy'])

        model.fit(train_text_matrix, train_labels,
                  batch_size=model_item.batch_size,
                  epochs=model_item.epochs,
                  validation_data=(dev_text_matrix, dev_labels),
                  callbacks=[checkpoint],
                  shuffle=True)
        model.load_weights(model_saved_path)
        
        test_loss, test_pre, test_rec, test_f_m, test_acc = model.evaluate(test_text_matrix, test_labels, batch_size=model_item.batch_size)
        dev_loss, dev_pre, dev_rec, dev_f_m, dev_acc = model.evaluate(dev_text_matrix, dev_labels, batch_size=model_item.batch_size)  
        
        pred_score = model.predict(test_text_matrix)      
        pred_test_label = np.argmax(pred_score, axis=1)
        
        
        test_dir = os.path.join(path_item.output_test_label,model_item.model_type)
        if os.path.exists(test_dir)==False:
            os.makedirs(test_dir)
        pred_test_f = open(os.path.join(test_dir, 'pred_label-' + str(idx)), 'w')
        pred_test_f.write('name pred_label\n')
        for name, p_label in zip(test_lists, pred_test_label):
            pred_test_f.write(name + ' ' + str(p_label) + '\n')
        pred_test_f.close()
        path_item.res_f.write(
                "test acc:" + str(round(test_acc, 4)) + "test rec:" + str(round(test_rec, 4))+ "test pre:" + str(round(test_pre, 4))+" test F-score:" + str(round(test_f_m, 4)) + '\n\n')
        test_accuracy.append(test_acc)
        test_f_score.append(test_f_m)
        test_percision.append(test_pre)
        test_recall.append(test_rec)
        
        
        val_accuracy.append(dev_acc)
        val_f_score.append(dev_f_m)        
        
        print('test_acc:'+str(test_acc)+'\n')
        print('test_f-score:'+str(test_f_m)+'\n')
        print('test_recall:'+str(test_rec)+'\n')
        print('test_precision:'+str(test_pre)+'\n')        
    val_accuracy = np.asarray(val_accuracy)
    val_accuracy = np.mean(val_accuracy, axis=0)
    val_f_score = np.asarray(val_f_score)
    val_f_score = np.mean(val_f_score, axis=0)
        
    test_accuracy = np.asarray(test_accuracy)
    test_accuracy = np.mean(test_accuracy, axis=0)
    test_f_score = np.asarray(test_f_score)
    test_f_score = np.mean(test_f_score, axis=0)
    test_percision = np.asarray(test_percision)
    test_percision = np.mean(test_percision, axis=0)
    test_recall = np.asarray(test_recall)
    test_recall = np.mean(test_recall, axis=0)
    print('test_accuracy:'+str(test_accuracy))
    print('test_f_score:'+str(test_f_score))
    print('test_recall:'+str(test_recall))
    print('test_percision:'+str(test_percision))
    
    print('val_accuracy:'+str(val_accuracy))
    print('val_f_score:'+str(val_f_score))
    
    path_item.res_f.write(
                "test acc:" + str(round(test_accuracy, 4)) + "test rec:" + str(round(test_recall, 4))+ "test pre:" + str(round(test_percision, 4))+" test F-score:" + str(round(test_f_score, 4)) + '\n\n')