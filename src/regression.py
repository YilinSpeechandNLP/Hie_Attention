import os
import pdb
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from src.functions import rmse
from src.data_load import glove_embedding_layer_gene,word2vec_embedding_layer_gene, test_data_load
from src.data_load import regression_data_load as load


def regression_func(trans_item,model_item,mmse_dict, path_item):

    
    dev_rmse = []
    test_rmse = []
    
    for idx in range(model_item.Fold):
        print('fold ' + str(idx))
        path_item.res_f.write('fold: ' + str(idx) + '\n')
        ### data_preparation
        train_lst_path = os.path.join(path_item.list_dir, str(idx), 'train_list')
        train_lists = open(train_lst_path).readlines()
        dev_lst_path = os.path.join(path_item.list_dir, str(idx), 'dev_list')
        dev_lists = open(dev_lst_path).readlines()
        test_lst_path = os.path.join(path_item.list_dir, str(idx), 'test_list')
        test_lists = open(test_lst_path).readlines()

        train_labels, train_texts = load(train_lists, path_item.train_transcript_path,mmse_dict)
        dev_labels, dev_texts = load(dev_lists, path_item.train_transcript_path,mmse_dict)
        test_labels, test_texts = load(test_lists, path_item.test_transcript_path,mmse_dict)

        all_texts = train_texts + dev_texts + test_texts
        tokenizer = Tokenizer(num_words=trans_item.max_NB_words)
        tokenizer.fit_on_texts(all_texts)
        word_index = tokenizer.word_index
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
            os.makedirs(model_folder_ex)
        model_saved_path = os.path.join(model_folder_ex, str(idx))
        model = model_item.model_select(trans_item,embedding_layer)

        checkpoint = ModelCheckpoint(model_saved_path, monitor='val_rmse', verbose=2, save_best_only=True,
                                         mode='min', save_weights_only=True)
        model.compile(loss='mse',
                          optimizer='adam',
                          metrics=[rmse])

        model.fit(train_text_matrix, train_labels,
                  batch_size=model_item.batch_size,
                  epochs=model_item.epochs,
                  validation_data=(dev_text_matrix, dev_labels),
                  callbacks=[checkpoint],
                  shuffle=True)
        model.load_weights(model_saved_path)
        
        dev_loss, dev_r = model.evaluate(dev_text_matrix, dev_labels, batch_size=model_item.batch_size)
        test_loss, test_r = model.evaluate(test_text_matrix, test_labels, batch_size=model_item.batch_size)
        
        pred_score = model.predict(test_text_matrix)     
        pred_score = pred_score*30 
        
        
        test_dir = os.path.join(path_item.output_test_label,model_item.model_type)
        if os.path.exists(test_dir)==False:
            os.makedirs(test_dir)
        pred_test_f = open(os.path.join(test_dir, 'pred_mmse-' + str(idx)), 'w')
        pred_test_f.write('name pred_mmse\n')
        for name, p_mmse in zip(test_lists, pred_score):
            pred_test_f.write(name + ' ' + str(p_mmse) + '\n')
        pred_test_f.close()
        
        
        
        
        
        
        dev_rmse.append(dev_r*30)
        test_rmse.append(test_r*30)
        
    dev_rmse = np.asarray(dev_rmse)
    dev_rmse_mean = np.mean(dev_rmse, axis=0)
    
    test_rmse = np.asarray(test_rmse)
    test_rmse_mean = np.mean(test_rmse, axis=0)

    print('dev_rmse:'+str(dev_rmse_mean)+'\n')
    print('test_rmse:'+str(test_rmse_mean)+'\n')
