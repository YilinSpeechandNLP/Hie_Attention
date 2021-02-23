import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec
from keras.models import Model
#from keras_bert import Tokenizer
#import keras
#from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs

### self defined functions
from src.functions import f1,rc,pr
from src.data_load import test_data_load, glove_embedding_layer_gene,word2vec_embedding_layer_gene
from src.data_load import classification_data_load as load
import pdb
def classification_attention_weights_extraction(trans_item,model_item,path_item):
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
        
        train_labels = to_categorical(np.asarray(train_labels), class_num)
        dev_labels = to_categorical(np.asarray(dev_labels), class_num)
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
        else:
            embedding_layer = glove_embedding_layer_gene(word_index, trans_item.embedding_len,path_item.glove_path)
        ## model initialization
        model_folder_ex = os.path.join(path_item.model_folder,model_item.model_type)
        if os.path.exists(model_folder_ex) == False:
            print('model doesn\'t exist\n')
            
        model_saved_path = os.path.join(model_folder_ex, str(idx))
        model = model_item.model_select(trans_item,embedding_layer)

        #checkpoint = ModelCheckpoint(model_saved_path, monitor='val_f1', verbose=2, save_best_only=True,
        #                                 mode='max', save_weights_only=True)
        #model.compile(loss='binary_crossentropy',
        #                  optimizer='adam',
        #                  metrics=[pr, rc, f1, 'accuracy'])
        model.load_weights(model_saved_path)
        all_lists = np.concatenate([train_lists, dev_lists, test_lists], axis = 0)
        text_matrix = np.concatenate([train_text_matrix, dev_text_matrix, test_text_matrix], axis = 0)
        
        att_output_model = Model(inputs=model.input,
                      outputs=model.get_layer('attention').output)
        
        attention_weights = att_output_model.predict(text_matrix, batch_size=model_item.batch_size)
        attention_weights = np.squeeze(attention_weights)
        weights_dict = {}
        for rec_info, att_weight in zip(all_lists, attention_weights):
            name = rec_info.split(' ')[0]
            weights_dict[name] = att_weight
        output_path = os.path.join(model_folder_ex, 'attention_weight_dict'+str(idx))  
        np.save(output_path, weights_dict) 