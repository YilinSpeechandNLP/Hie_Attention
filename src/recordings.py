import sys


def recording(transcript_item,model_item,path_item):
    print('#############################################')
    print('parameters format: training_type model_type')
    print('training type: classification\t regression')
    print('model type: hie_attention\t hie\t bi-lstm\t bi-lstm-attention')
    print('for example: python main.py classification Hie')
    print('input parameters number: ' + str(len(sys.argv)) + '\n')
    path_item.res_f.write('parameters:\n********************************\n')
    path_item.res_f.write('training_type:' + model_item.train_type + '\n')
    path_item.res_f.write('MAX_SENT_LENGTH:' + str(transcript_item.word_num_per_sent) + '\n')
    path_item.res_f.write('MAX_SENTS:' + str(transcript_item.sent_nums) + '\n')
    path_item.res_f.write('MAX_NB_WORDS:' + str(transcript_item.max_NB_words) + '\n')
    path_item.res_f.write('LSTM_DIM:' + str(model_item.LSTM_DIM) + '\n')
    path_item.res_f.write('DENSE_DIM:' + str(model_item.Dense_DIM) + '\n')
    path_item.res_f.write('batch_size:' + str(model_item.batch_size) + '\n')
    path_item.res_f.write('Drop_rate:' + str(model_item.drop_rate) + '\n')
    path_item.res_f.write('Att_DIM:' + str(model_item.Att_DIM) + '\n')
    path_item.res_f.write('epochs:' + str(model_item.epochs) + '\n')
    path_item.res_f.write('L2 regularizers' + str(model_item.l2) + '\n')
    path_item.res_f.write('********************************\n')


