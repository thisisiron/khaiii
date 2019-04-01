
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, TimeDistributed, Bidirectional, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

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

