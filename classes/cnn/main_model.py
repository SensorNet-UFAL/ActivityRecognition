#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#############################################################################################

from alstm import *

def cnn_model_simple(input_shape):
    model = Sequential()
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    
    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 3,input_shape=input_shape))
    model.add(Convolution1D(32, 3))
    model.add(Convolution1D(16, 3))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(size_class,activation='sigmoid'))
    return model

#############################################################################################

#############################################################################################

def cnn_model_large(input_shape):

    # Convolutional model (3x conv, 1x maxP, 2x conv, 1x maxP, 3x conv, 1x maxP, flatten, 2x dense, 1x droup, 2x dense)
    model = Sequential()
#    model.add(BatchNormalization())
    model.add(Convolution1D(128, 3, input_shape = input_shape))
    model.add(Convolution1D(128, 3))
    model.add(Convolution1D(128, 3))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Convolution1D(128, 4))
    model.add(Convolution1D(128, 4))
    model.add(Convolution1D(128, 4))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(2))
    model.add(Convolution1D(64, 5))
    model.add(Convolution1D(64, 5))
    model.add(Convolution1D(64, 5))
    model.add(BatchNormalization())

#    model.add(MaxPooling1D(2))
#    model.add(Convolution1D(32, 5))
#    model.add(Convolution1D(32, 5))
#    model.add(Convolution1D(32, 5))
#    model.add(BatchNormalization())
    model.add(LSTM(300, return_sequences=True))
#    model.add(LSTM(200, return_sequences=True))
#    model.add(LSTM(100, return_sequences=True))
    #model.add(Flatten())

#    model_aux = Sequential()
#    model_aux.add(Dense(1024, input_shape = (18, ),activation = 'relu'))
#    model_aux.add(Dense(12, activation = 'relu'))
#    model_aux.add(BatchNormalization())

#    model.add(Dropout(0.2))
#    model.add(Dense(1024,activation='relu'))
 #   model.add(BatchNormalization())

#    model_final = Sequential()
#    model_final.add(Merge([model, model_aux], mode = 'concat'))

#    model_final.add(Dense(1024,activation='relu'))
    #model.add(BatchNormalization())

#    model_final.add(Dropout(0.2))
 #   model_final.add(Dense(1024,activation='relu'))
  #  model_final.add(Dropout(0.2))
 #   model_final.add(Dense(256,activation='relu'))
    #model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(30,activation='sigmoid')))
    model.add(Dense(size_class,activation='sigmoid'))    
    return model#_final

#############################################################################################

def rnn_main_model(input_shape):
    model = Sequential()
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 3,input_shape=input_shape))
    model.add(Convolution1D(32, 3))
    model.add(Convolution1D(16, 3))
    model.add(MaxPooling1D(2))
    model.add(LSTM(100))
    #model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(size_class,activation='sigmoid'))
    return model


def generate_model(input_shape):
    ip = Input(shape=input_shape)

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(size_class, activation='softmax')(x)

    model = Model(ip, out)

    return model

def generate_model_2(input_shape):
    ip = Input(shape=input_shape)
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(size_class, activation='softmax')(x)

    model = Model(ip, out)


    # add load model code here to fine-tune

    return model

def generate_model_3(input_shape):
    ip = Input(shape=input_shape)

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(size_class, activation='softmax')(x)

    model = Model(ip, out)


    # add load model code here to fine-tune

    return model
    
    
def generate_model_4(input_shape):
    ip = Input(shape=input_shape)
    # stride = 3
    #
    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(size_class, activation='softmax')(x)

    model = Model(ip, out)

    # add load model code here to fine-tune

    return model

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
