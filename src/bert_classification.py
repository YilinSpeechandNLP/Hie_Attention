import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec
from tensorflow.keras import layers
from keras import backend as K
from tensorflow import keras
import pdb
### self defined functions
from src.functions import rc,pr
from src.data_load import classification_data_load as load

import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import regularizers, initializers
import keras.backend as K

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val




class AttLayer(layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        super(AttLayer, self).__init__(**kwargs)
        self.init = initializers.get('normal')
        self.attention_dim = attention_dim
         
          
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = tf.Variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = tf.Variable(self.init((self.attention_dim, )))
        self.u = tf.Variable(self.init((self.attention_dim, 1)))
        self.epsilon = tf.constant(value=0.000001, shape=input_shape[1])
        super(AttLayer, self).build(input_shape)
        
    def compute_mask(self, inputs, mask=None):
        return mask        
        
        
    def call(self, inputs, mask=None):
        uit = tf.math.tanh(tf.nn.bias_add(tf.matmul(inputs, self.W), self.b))
        ait = tf.matmul(uit, self.u)
        ait = tf.squeeze(ait, -1)
        ait = tf.math.exp(ait)
        ait /= tf.cast(tf.math.reduce_sum(ait, axis=1, keepdims=True) + self.epsilon, dtype=tf.float32)
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait
        output = tf.math.reduce_sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
    def build(self, input_shape):   
        
        super(TransformerBlock, self).build(input_shape)  # Be sure to call this somewhere!     
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_dim)    


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, trainable=True)
        self.embed_dim = embed_dim
    def build(self, input_shape):   
        super(TokenAndPositionEmbedding, self).build(input_shape)  # Be sure to call this somewhere!     
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_dim)

def BERT_embedding_layer_gene(maxlen, vocab_size, embed_dim):
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    transformer_block = TransformerBlock(embed_dim, num_heads=2, ff_dim=32)
    return [embedding_layer,transformer_block]
    
    
def BERT_Att_model(model_item, trans_item, embedding_layer,transformer_block):
    sequence_input = layers.Input(shape=(trans_item.word_num_per_trans,), dtype='float32')
    x = embedding_layer(sequence_input)
    x = transformer_block(x)
    x = layers.Bidirectional(layers.LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = layers.Dropout(model_item.drop_rate)(x)
    x = AttLayer(model_item.Att_DIM,name = 'sent_attention')(x)
    #x = layers.Dense(model_item.Dense_DIM, activation='relu',name = 'dense')(x)
    preds = layers.Dense(model_item.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(model_item.l2))(x)
    model = keras.Model(sequence_input, preds)
    print(model.summary())
    return model
    
def BERT_model(model_item, trans_item, embedding_layer,transformer_block):
    sequence_input = layers.Input(shape=(trans_item.word_num_per_trans,), dtype='float32')
    x = embedding_layer(sequence_input)
    x = transformer_block(x)
    x = layers.Bidirectional(layers.LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(model_item.Dense_DIM, activation='relu',name = 'dense')(x)
    preds = layers.Dense(model_item.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(model_item.l2))(x)
    model = keras.Model(sequence_input, preds)
    print(model.summary())
    return model

def Hie_BERT_model(model_item, trans_item, embedding_layer, transformer_block):
    ############word-level#############
    sentence_input = layers.Input(shape=(trans_item.word_num_per_sent,), dtype='int32')
    x = embedding_layer(sentence_input)
    x = transformer_block(x)
    x = layers.Bidirectional(layers.LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = layers.Dropout(model_item.drop_rate)(x)
    x = layers.Dense(model_item.Dense_DIM, activation='relu')(x)
    output = AttLayer(model_item.Att_DIM,name = 'word_attention')(x)
    sentEncoder = keras.Model(sentence_input, output)
    print(sentEncoder.summary())
    
    ############sentence-level: Bi-rnn############
    review_input = layers.Input(shape=(trans_item.sent_nums, trans_item.word_num_per_sent), dtype='int32')
    x = layers.TimeDistributed(sentEncoder)(review_input)
    x = layers.Bidirectional(layers.LSTM(model_item.LSTM_DIM, return_sequences=True), merge_mode='sum')(x)
    x = layers.Dropout(model_item.drop_rate)(x)
    x = AttLayer(model_item.Att_DIM,name = 'sent_attention')(x)
    preds = layers.Dense(model_item.label_num, activation='sigmoid', kernel_regularizer=regularizers.l2(model_item.l2))(x)
    model = keras.Model(review_input, preds)
    print(model.summary())
    return model
def bert_classification_func(trans_item,model_item,path_item):

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

        embedding_layer,transformer_block = BERT_embedding_layer_gene(trans_item.word_num_per_trans, trans_item.max_NB_words, trans_item.embedding_size)
        ## model initialization
          
        model_folder_ex = os.path.join(path_item.model_folder,model_item.model_type)
        if os.path.exists(model_folder_ex) == False:
            os.makedirs(model_folder_ex)
   
        model_saved_path = os.path.join(model_folder_ex, str(idx))
        model = BERT_Att_model(model_item, trans_item, embedding_layer,transformer_block)
        checkpoint = ModelCheckpoint(model_saved_path, monitor='val_f1', verbose=2, save_best_only=True,
                                         mode='max', save_weights_only=True)
        model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[pr, rc, f1,'accuracy'])
        
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