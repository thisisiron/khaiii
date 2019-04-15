<<<<<<< HEAD
=======
# -*- coding: utf-8 -*-


###########
# imports #
###########
>>>>>>> b7b3300e638d76daddc71e13bd70d13b7dd8dc86

import numpy as np
import tensorflow as tf

<<<<<<< HEAD
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, TimeDistributed, Bidirectional, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

=======
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, TimeDistributed, Bidirectional, LSTM, Add
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


#############
# functions #
#############
>>>>>>> b7b3300e638d76daddc71e13bd70d13b7dd8dc86
def to_categorical(sequences, num_categories):
    cat_sequences = [] 
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(num_categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
       y_true_class = K.argmax(y_true, axis=-1)
       y_pred_class = K.argmax(y_pred, axis=-1)
                        
       ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
       matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
       accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
       return accuracy
    return ignore_accuracy


<<<<<<< HEAD

def model(cfg, rsc):
    embedding_size = cfg.embedding_size
    input_size = cfg.input_length 
    num_of_classes = len(rsc.vocab_out.dic)

    inputs = Input(shape=(input_size,), name='input')
    x = Embedding(len(rsc.vocab_in.dic)+1, embedding_size, input_length=input_size)(inputs)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    outputs = TimeDistributed(Dense(num_of_classes, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', ignore_class_accuracy(0)])
    model.summary()
    return model


class BiLSTMModel():
    def __init__(self, cfg, rsc):
        self.cfg = cfg
        self.rsc = rsc
        self.embedding_size = self.cfg.embedding_size
        self.sequence_len = self.cfg.input_length
        self.num_of_classes = len(self.rsc.vocab_out.dic)

    def model(self):
        inputs = Input(shape=(self.sequence_len,), name='input')
        x = Embedding(len(self.rsc.vocab_in.dic)+1, self.embedding_size, input_length=self.sequence_len)(inputs)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        outputs = TimeDistributed(Dense(self.num_of_classes, activation='softmax'))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', ignore_class_accuracy(0)])
        model.summary()
        return model

=======
class Embedder(tf.keras.Model):
    def __init__(self, embedding_size, input_length, vocab_size, using_pos=True):
        super(Embedder, self).__init__(name='')
        self.using_pos = using_pos
        self.token_emb = Embedding(vocab_size, embedding_size, input_length=input_length)
        if using_pos:
            self.pos_emb = Embedding(input_length, embedding_size, trainable=False, input_length=input_length, weights=[self._get_pos_embedding(input_length, embedding_size)])
        self.sum_emb = Add()

    def call(self, input_tensor):
        x = self.token_emb(input_tensor)
        if self.using_pos:
            y = self.pos_emb(input_tensor)
            return self.sum_emb([x,y])
        else:
            return x

    def _get_pos_embedding(self, max_len, dim_emb):
        pos_emb = np.array(
            [[pos / np.power(10000, 2 * (j//2) / dim_emb) for j in range(dim_emb)] if pos != 0 else np.zeros(dim_emb) \
              for pos in range(max_len)]
        )
        pos_emb[1:, 0::2] = np.sin(pos_emb[1:, 0::2])
        pos_emb[1:, 1::2] = np.cos(pos_emb[1:, 1::2])
        return pos_emb




#class ELMo(tf.keras.Model):
#    def __init__(self, hidden_units, ):
#        super(ELMo, self).__init__(name=='')
#
#        self.left_lstm_1 = LSTM(hidden_units, return_sequences=True, )
#        self.left_lstm_2 = LSTM(hidden_units, return_sequences=True, )
#
#
#        self.right_lstm_1 = LSTM(hidden_units, return_sequences=True,)
#        self.right_lstm_2 = LSTM(hidden_units, return_sequences=True, )
        


def model(cfg, rsc):
    embedding_size = cfg.embedding_size
    input_length = cfg.input_length 
    vocab_size = len(rsc.vocab_in.dic)+1
    num_of_classes = len(rsc.vocab_out.dic)

    inputs = Input(shape=(input_length,), name='input')
    embedder = Embedder(embedding_size, input_length, vocab_size)
    x = embedder(inputs)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    outputs = TimeDistributed(Dense(num_of_classes, activation='softmax'))(x)
    
    return Model(inputs=inputs, outputs=outputs)
>>>>>>> b7b3300e638d76daddc71e13bd70d13b7dd8dc86
