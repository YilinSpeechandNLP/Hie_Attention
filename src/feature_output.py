### output the trained attention layer extracted features
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from src.data_load import classification_data_load, regression_data_load,test_data_load,embedding_layer_gene


def feature_output(trans_item,model_item,path_item):
    test_lists = os.listdir(path_item.test_transcript_path)
    test_texts = test_data_load(test_lists, path_item.test_transcript_path)
    for idx in range(model_item.Fold):
        print('fold  ' + str(idx))
        train_lst_path = os.path.join(path_item.list_dir, str(idx), 'trans_train')
        train_lists = open(train_lst_path).readlines()
        dev_lst_path = os.path.join(path_item.list_dir, str(idx), 'trans_dev')
        dev_lists = open(dev_lst_path).readlines()
        if trans_item.train_type == 'classification':
            train_labels, train_texts = classification_data_load(train_lists, path_item.transcript_path)
            dev_labels, dev_texts = classification_data_load(dev_lists, path_item.transcript_path)
        else:
            mmse_dict = np.load(path_item.mmst_dict_path, allow_pickle=True).item()
            train_labels, train_texts = regression_data_load(train_lists, path_item.transcript_path,mmse_dict)
            dev_labels, dev_texts = regression_data_load(dev_lists, path_item.transcript_path,mmse_dict)

        all_texts = train_texts + dev_texts + test_texts
        tokenizer = Tokenizer(num_words=trans_item.max_NB_words)
        tokenizer.fit_on_texts(all_texts)
        word_index = tokenizer.word_index

        # padding: regular the data format into the requirement matrix
        train_text_matrix = trans_item.matrix_gene(trans_item.model_type,train_texts, tokenizer)
        dev_text_matrix =trans_item.matrix_gene(trans_item.model_type,dev_texts, tokenizer)
        test_text_matrix = trans_item.matrix_gene(trans_item.model_type,test_texts, tokenizer)

        # development set
        path = os.path.join(path_item.model_folder,model_item.model_type,str(idx))
        filepath = path + "/best_weights.h5"
        embedding_layer = embedding_layer_gene(word_index, trans_item.embedding_len, path_item.glove_path)
        model = model_item.model_select(trans_item, embedding_layer)
        model.load_weights(filepath)
        ## define the model for extracting the trained feature
        if model_item.model_type == 'Hie_Att' or model_item.model_type == 'BiRNN_Att':
            feats_extract_model = Model(inputs=model.input, outputs=model.get_layer('attention' ).output)
        else:
            feats_extract_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)

        ## feature extraction
        train_extracted_feats = feats_extract_model.predict(train_text_matrix)
        dev_extracted_feats = feats_extract_model.predict(train_text_matrix)
        test_extracted_feats = feats_extract_model.predict(train_text_matrix)

        ## folder generate
        feats_folder = os.path.join(path_item.feats_extract_folder,model_item.model_type)
        output_path = os.path.join(feats_folder, 'pred_label-' + str(idx))

        # feature output
        train_dict = {}
        for name, att_vector in zip(train_lists, train_extracted_feats):
            train_dict[name.split(' ')[0]] = att_vector
        np.save(os.path.join(output_path,'train'), train_dict)

        dev_dict = {}
        for name, att_vector in zip(dev_lists, dev_extracted_feats):
            dev_dict[name.split(' ')[0]] = att_vector
        np.save(os.path.join(output_path,'dev'), dev_dict)

        test_dict = {}
        for name, att_vector in zip(test_lists, test_extracted_feats):
            test_dict[name.split(' ')[0]] = att_vector
        np.save(os.path.join(output_path,'test'), test_dict)