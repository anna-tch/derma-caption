
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

import tensorflow as tf
import keras.backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils import plot_model
from keras import optimizers
from keras.layers import RepeatVector
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM ,GRU
from keras.layers import Embedding
from keras.layers import Dropout, Reshape, Lambda, Concatenate
from keras.layers.merge import add





def data_generator(descriptions, photos, wordtoix, max_length, batch_size=32):
    X1, X2, y = [], [], []
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # find the index of each word of the caption in vocabulary
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                # Each step of the following for loop selects one word
                # from the caption, consider that word as y and
                # all the words before that will be the X
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i] # words until i are inseq word i is outseq
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                     out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == batch_size:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0


def masked_loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits= False)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def make_model(embedding ,  max_length, vocab_size, dout= 0.2, feature_size= 2048, units= 256):

    # output size of feature extractor
    features = Input(shape=(feature_size,))
    # because i have used bidirectional LSTM, the number of units should
    # become double here in order for the add function to work
    X_fe_one_dim = Dense(units, activation='relu')(features)
    X_fe = RepeatVector(max_length)(X_fe_one_dim)
    X_fe = Dropout(dout)(X_fe)

    seq = Input(shape=(max_length,))
    X_seq = embedding(seq)
    # remove mask from the embedding cause concat doesn't support it
    X_seq = Lambda(lambda x: x, output_shape=lambda s:s)(X_seq)
    X_seq = Dropout(dout)(X_seq)
    X_seq = Concatenate(name='concat_features_word_embeddings', axis=-1)([X_fe,X_seq])
    # passing features as init_state
    X_seq = GRU(units, return_sequences=True)(X_seq,initial_state=X_fe_one_dim)
    X_seq = Dropout(dout + 0.2)(X_seq)
    X_seq = GRU(units, return_sequences=False)(X_seq)

    # decode
    outputs = Dense(vocab_size, activation='softmax')(X_seq)

    # merge the two input models
    model = Model(inputs=[features, seq], outputs = outputs, name='Image captioning model')
    print(model.summary())
    plot_model(model, to_file='../output/model.png', show_shapes=True)


    return model
