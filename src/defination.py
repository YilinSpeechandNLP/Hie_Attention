from keras.engine.topology import Layer
from keras import initializers
import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Dense, Input, Flatten, Lambda
from keras.layers import  Dropout, GRU, Bidirectional, TimeDistributed, LSTM
from keras.models import Model
from keras import regularizers
import os
import pdb

class path_setup_item:
    def __init__(self, train_type,train_transcript_path,test_transcript_path,data_folder):
    
        self.train_transcript_path = train_transcript_path
        self.test_transcript_path = test_transcript_path
        self.list_dir = os.path.join('/data/ac1yp/data/cookie_theft/lists',data_folder,'list')
        self.test_list = os.path.join('/data/ac1yp/data/cookie_theft/lists',data_folder,'list')
        self.output_test_label = os.path.join('output', data_folder, 'pred/pred_labels/test')
        self.output_dev_label = os.path.join('output', data_folder, 'pred/pred_labels/dev')
        self.output_test_score = os.path.join('output', data_folder,'pred/pred_mmse/test')
        self.output_dev_score = os.path.join('output',data_folder,'pred/pred_mmse/dev')
        self.model_folder = os.path.join('output/trained_models', data_folder)
        if os.path.exists(self.model_folder) == False:
            os.makedirs(self.model_folder)
        self.glove_path = '/fastdata/ac1yp/pre_trained_word_embedding/glove/glove.6B.100d.txt'
        self.word2vec_path= '/data/ac1yp/code/Challenge_TAPAS/output/word2vec.model'
        self.label_dict_path =os.path.join('/data/ac1yp/data/cookie_theft/label_dict.npy')
        self.mmst_dict_path ='/data/ac1yp/code/Challenge_TAPAS/dataset/text/mmse.npy'
        self.res_record_path = os.path.join('output',data_folder,'res_record/')
        self.feats_extract_folder = os.path.join('output',data_folder,'feats/')
        output_res_path = os.path.join(self.res_record_path, train_type)
        if os.path.exists(output_res_path) == False:
            os.makedirs(output_res_path)
            ## save the parameters ahead
        self.res_f = open(os.path.join(output_res_path, 'res.txt'), 'a+')

class transcript_setup_item:
    def __init__(self, model_type, sent_nums = 30,word_num_per_sent = 30, word_num_per_trans = 200, max_NB_words = 1500):
        self.sent_nums = sent_nums
        self.word_num_per_sent = word_num_per_sent
        self.word_num_per_trans = word_num_per_trans
        self.max_NB_words = max_NB_words
        if model_type == 'Hie' or model_type == 'Hie_Att':
            self.embedding_len = self.word_num_per_sent
        else:
            self.embedding_len = self.word_num_per_trans
        self.embedding_size = 100
    def hierarchical_matrix_gen(self,texts,tokenizer):
        data = np.zeros((len(texts), self.sent_nums, self.word_num_per_sent), dtype='int32')
        for i, text in enumerate(texts):
            sentences = tokenize.sent_tokenize(text)
            for j, sent in enumerate(sentences):
                if j < self.sent_nums:
                    wordTokens = text_to_word_sequence(sent)
                    for k, word in enumerate(wordTokens):
                        if k < self.word_num_per_sent and tokenizer.word_index[word] < self.max_NB_words:
                            data[i, j, k] = tokenizer.word_index[word]
        print(('Shape of data tensor:', data.shape))
        return data
    def single_layer_matrix_gene(self,texts,tokenizer):
        # development set
        text = tokenizer.texts_to_sequences(texts)
        text = pad_sequences(text, maxlen=self.word_num_per_trans)
        return text
    def matrix_gene(self,model_type,texts,tokenizer):
        if model_type == 'Hie' or model_type =='Hie_Att':
            matrix = self.hierarchical_matrix_gen(texts,tokenizer)
            return matrix
        elif model_type == 'BiRNN' or model_type == 'BiRNN_Att':
            matrix = self.single_layer_matrix_gene(texts,tokenizer)
            return matrix
        else:
            print('wrong model type!\n')
            exit(0)



class model_setup_item:
    def __init__(self,  model_type, train_type, batch_size = 10,drop_rate = 0.3,
                 epochs = 20,l2 = 0.0001, Att_DIM = 30, Dense_DIM = 50, LSTM_DIM = 100, Fold = 10):
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.l2 = l2
        self.Att_DIM = Att_DIM
        self.Dense_DIM = Dense_DIM
        self.LSTM_DIM = LSTM_DIM
        self.Fold = Fold
        self.train_type = train_type
        self.model_type = model_type
        
        if self.train_type == 'classification' or self.train_type == 'classification_attention_extraction' :
            self.label_num = 2
            self.class_num = 2
        else:
            self.label_num = 1
        self.embedding_type = 'BERT'
    def Hie_Att_model_building(self,trans_item,embedding_layer):
        ############word-level#############
        sentence_input = Input(shape=(trans_item.word_num_per_sent,), dtype='int32')
        x = embedding_layer(sentence_input)
        x = Dropout(self.drop_rate)(x)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.Dense_DIM, activation='relu')(x)
        output = AttLayer(self.Att_DIM,name = 'word_attention')(x)
        sentEncoder = Model(sentence_input, output)
        print(sentEncoder.summary())
        ############sentence-level: Bi-rnn############
        review_input = Input(shape=(trans_item.sent_nums, trans_item.word_num_per_sent), dtype='int32')
        x = TimeDistributed(sentEncoder)(review_input)
        x = Bidirectional(GRU(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        x = AttLayer(self.Att_DIM,name = 'sent_attention')(x)
        preds = Dense(self.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(review_input, preds)
        print(model.summary())
        return model
    def Hie_model_building(self, trans_item,embedding_layer):
        ############word-level#############
        sentence_input = Input(shape=(trans_item.word_num_per_sent,), dtype='int32')
        x = embedding_layer(sentence_input)
        x = Dropout(self.drop_rate)(x)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        x = Flatten()(x)
        output = Dense(self.Dense_DIM, activation='relu')(x)
        sentEncoder = Model(sentence_input, output)
        print(sentEncoder.summary())
        ############sentence-level: Bi-rnn############
        review_input = Input(shape=(trans_item.sent_nums, trans_item.word_num_per_sent), dtype='int32')
        x = TimeDistributed(sentEncoder)(review_input)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        #x = Dropout(self.drop_rate)(x)
        x = Flatten()(x)
        x = Dense (self.Dense_DIM, activation='relu',name = 'dense')(x)
        preds = Dense(self.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(review_input, preds)
        print(model.summary())
        return model
    def BiRNN_model_building(self,trans_item,embedding_layer):

        sequence_input = Input(shape=(trans_item.word_num_per_trans,), dtype='int32')
        x = embedding_layer(sequence_input)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Flatten()(x)
        x = Dense(self.Dense_DIM, activation='relu',name = 'dense')(x)
        preds = Dense(self.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(sequence_input, preds)
        print(model.summary())
        return model

    def BiRNN_Att_Model_building(self,trans_item,embedding_layer):
        sequence_input = Input(shape=(trans_item.word_num_per_trans,), dtype='int32')
        x = embedding_layer(sequence_input)

        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        x = AttLayer(self.Att_DIM, name = 'attention')(x)
        preds = Dense(self.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(sequence_input, preds)
        print(model.summary())
        return model
    def model_select(self,trans_item, embedding_layer):
        if self.model_type == 'Hie':
            model = self.Hie_model_building(trans_item,embedding_layer)
            return model
        elif self.model_type == 'Hie_Att':
            model = self.Hie_Att_model_building(trans_item,embedding_layer)
            return model
        elif self.model_type == 'BiRNN':
            model = self.BiRNN_model_building(trans_item,embedding_layer)
            return model
        elif self.model_type == 'BiRNN_Att':
            pdb.set_trace()
            if len(embedding_layer) == 2:
                model = self.BERT_Model_building(trans_item,embedding_layer)
            else:
                model = self.BiRNN_Att_Model_building(trans_item,embedding_layer)
            return model
        else:
            print('model type is wrong!\n')
            exit(0)

class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
'''
class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)

        #weighted_input = x * ait
        #output = K.sum(weighted_input, axis=1)

        return ait

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)
'''
'''
    def Hie_Att_model_building(self,trans_item,embedding_layer):
        ############word-level#############
        sentence_input = Input(shape=(trans_item.word_num_per_sent,), dtype='int32')
        x = embedding_layer(sentence_input)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.Dense_DIM, activation='relu')(x)
        ait_word = AttLayer(self.Att_DIM,name='word_attention')(x)
        sentWeight = Model(sentence_input, ait_word)
        sentEncoder = Model(sentence_input, x)
        print(sentWeight.summary())
        print(sentEncoder.summary())
        ############sentence-level: Bi-rnn############
        review_input = Input(shape=(trans_item.sent_nums, trans_item.word_num_per_sent), dtype='int32')
        Sent_D = TimeDistributed(sentEncoder)(review_input)
        weight_D = TimeDistributed(sentWeight, name = 'word_weight')(review_input)
        my_Sum = Lambda(lambda x: K.sum(np.multiply(x[0], x[1]), axis=1))
        Time_D = my_Sum([Sent_D, weight_D])
        #x = TimeDistributed(sentEncoder)(review_input)
        x = Bidirectional(GRU(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(Time_D)
        #x = Dropout(self.drop_rate)(x)
        att_sent = AttLayer(self.Att_DIM,name = 'sent_attention')(x)
        x = my_Sum([x, att_sent])
        preds = Dense(self.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(review_input, preds)
        print(model.summary())
        return model

'''
'''
    def BiRNN_Att_Model_building(self,trans_item,embedding_layer):
        sequence_input = Input(shape=(trans_item.word_num_per_trans,), dtype='int32')
        x = embedding_layer(sequence_input)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        my_Sum = Lambda(lambda x: K.sum(np.multiply(x[0], x[1]), axis=1))
        att_sent = AttLayer(self.Att_DIM, name = 'attention')(x)
        x = my_Sum([x, att_sent])
        preds = Dense(self.label_num, activation='softmax')(x)
        model = Model(sequence_input, preds)
        print(model.summary())
        return model
'''

'''
    def BERT_Model_building(self,trans_item,embedding_block):
        pdb.set_trace()
        sequence_input = Input(shape=(trans_item.word_num_per_trans,), dtype='int32')
        [embedding_layer,transformer_block] = embedding_block
        x = embedding_layer(sequence_input)
        x = transformer_block(x)
        x = Bidirectional(LSTM(self.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
        x = Dropout(self.drop_rate)(x)
        x = AttLayer(self.Att_DIM, name = 'attention')(x)
        preds = Dense(self.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2))(x)
        model = Model(sequence_input, preds)
        print(model.summary())
        return model
'''