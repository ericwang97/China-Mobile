from sklearn import preprocessing
import numpy as np
import pandas as pd
import keras as K
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
from keras import optimizers
from sklearn import metrics
import tensorflow as tf

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def NN_processing(X_train,X_test,Y_train,Y_test):


    X_train_NN = preprocessing.scale(X_train).astype('float32')  # 归一化
    X_test_NN = preprocessing.scale(X_test).astype('float32')
    Y_train_NN = Y_train.astype('float32')
    Y_test_NN = Y_test.astype('float32')

    return X_train_NN,X_test_NN,Y_train_NN,Y_test_NN

def NN_main_func(original_dim,intermediate_dim,loss,dropout,activation,output_activation):

    model = Sequential()
    model.add(Dense(original_dim, input_dim=original_dim, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(intermediate_dim[0], activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(intermediate_dim[1], activation=activation))
    model.add(Dense(1, activation=output_activation))
    model.compile(optimizer=optimizers.adam(lr=0.006),
                  loss=loss,
                  metrics=['accuracy'])
    print(model.summary())

    return model
