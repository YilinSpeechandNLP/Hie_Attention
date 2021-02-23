import sys
from src.classification import classification_func
from src.bert_classification import bert_classification_func
from src.regression import regression_func
from src.classification_word2vec import word2vec_classification_func
from src.regression_word2vec import word2vec_regression_func
from src.attention_weights_extraction import classification_attention_weights_extraction
import numpy as np
import pdb
from src.recordings import recording
from src.defination import model_setup_item,transcript_setup_item,path_setup_item
def main():
    ### input paramters
    train_type = sys.argv[1]
    model_type = sys.argv[2]
    train_transcript_path = sys.argv[3]
    test_transcript_path = sys.argv[3]
    data_folder = sys.argv[4]

    ## initialized  items
    transcript_item = transcript_setup_item(model_type)
    model_item = model_setup_item(model_type, train_type)
    path_item = path_setup_item(train_type, train_transcript_path, test_transcript_path,data_folder)
    # recording the parameters in the text
    recording(transcript_item,model_item,path_item)
    if model_item.embedding_type == 'BERT':
        if train_type == 'regression':
            mmse_dict = np.load(path_item.mmst_dict_path, allow_pickle=True).item()
            #bert_regression_func(transcript_item, model_item, mmse_dict, path_item)
        elif train_type == 'classification':
            print('embedding type should be BERT')
            bert_classification_func(transcript_item,model_item,path_item)
        else:
            print('training type wrong!\n')
            exit(0)
    else:
        if train_type == 'regression':
            mmse_dict = np.load(path_item.mmst_dict_path, allow_pickle=True).item()
            regression_func(transcript_item, model_item, mmse_dict, path_item)
    
        elif train_type == 'classification':
            classification_func(transcript_item,model_item,path_item)
        elif train_type == 'classification_attention_extraction':
            classification_attention_weights_extraction(transcript_item,model_item,path_item)
        else:
            print('training type wrong!\n')
            exit(0)

if __name__ == "__main__":
    main()