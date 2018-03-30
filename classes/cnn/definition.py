#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, MaxPooling1D, Merge, Input, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.preprocessing import sequence
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Recurrent
from alstm import *

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

import time

import pandas as pd
from os import listdir
from os.path import isfile, join
from random import shuffle

from sklearn import preprocessing

import tqdm
from tqdm import *
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import iqr
from statsmodels import robust

execfile('prepare.py')
execfile('train.py')
execfile('predict.py')
execfile('main_model.py')

