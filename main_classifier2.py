#!/usr/bin/python

import numpy as np
import os
import sys
import tensorflow as tf
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Input, merge
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, Merge
from keras.optimizers import SGD, RMSprop, Adam, Nadam, Adadelta
from keras.callbacks import *
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
import random
import pickle
from gensim.models import Word2Vec
from tabulate import tabulate
import csv

sess = tf.Session()
K.set_session(sess)
###################################################################

MAX_SENT_LEN = 100
W2V = Word2Vec.wv.load('wordvectors.w2v.model')
X_sent_train = []
X_word_train = []
X_connect_train = []
Y_train = []

X_train_sent = []
X_train_words = []
X_connect_words = []

def clean(s):
    s = s.split(' ')
    s = s[:100]
    while(len(s) < 100):
        s.append('<PAD>')
    return s

def getWordVec(w):
    try:
        res = W2V[s]
    except Exception:
        res = [0] * 100
    return res

def getSentVec(s):
    res = []
    for i in s:
        res.append(getWordVec(i))
    return res

def toCategorical(label):
	res = [0, 0, 0]
	res[label] = 1
	return res

def readData(fname):
    global X_train_sent
    global X_train_words
    global Y_train_words
    csvfile = open(fname,'rb')
    for j, row in enumerate(csv.reader(csvfile, delimiter = ',', quotechar='"')):
        curVecSent = []
        curVecWords = []
        if(len(row) >= 2):
            sent = clean(row[0])
            curVecSent = getSentVec(sent)
            words = row[1]
            connect = row[2]
            label = int(row[3])
            X_sent_train.append(curVecSent)
            X_word_train.append(getWordVec(words))

            X_train_sent.append(sent)
            X_train_words.append(words)

            X_connect_words.append(connect)
            X_connect_train.append(getWordVec(connect))

            Y_train.append(toCategorical(label))
            

def train():
    input_a = Input(shape = (MAX_SENT_LEN,100,))
    input_b = Input(shape = (100,))
    input_c = Input(shape = (100,))

    sentNetwork = Sequential()
    sentNetwork.add(LSTM(100, return_sequences = False, input_shape = (MAX_SENT_LEN, 100, )))
    sentNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))
    sentNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))

    wordNetwork = Sequential()
    wordNetwork.add(Dense(100, input_shape=(100,), init = 'glorot_normal', activation = 'tanh'))
    wordNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))
    wordNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))

    connectNetwork = Sequential()
    connectNetwork.add(Dense(100, input_shape=(100,), init = 'glorot_normal', activation = 'tanh'))
    connectNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))
    connectNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))

    mainNetwork = Sequential()
    mainNetwork.add(Merge([sentNetwork, wordNetwork, connectNetwork], mode='concat'))
    mainNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))
    mainNetwork.add(Dense(100, init = 'glorot_normal', activation = 'tanh'))
    mainNetwork.add(Dense(3, init = 'glorot_normal', activation = 'softmax')) # part of arg1/part of arg2/neither

    probs = mainNetwork([input_a, input_b, input_c])
    model = Model(input = [input_a, input_b, input_c], output = [probs])
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    model.fit([X_sent_train, X_word_train, X_connect_train], [Y_train], batch_size = 64, nb_epoch = 20, verbose = 2)
    return model

def predictOnTestSet(model):
	X_sent_test = []
	X_word_test = []
	X_connect_test = []
	probs = model.predict(X_sent_test, X_word_test, X_connect_test)

fname = ''
readData(fname)
model = train()
predictOnTestSet(model)