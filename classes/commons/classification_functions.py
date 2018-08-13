#IMPORTS#
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from classes.commons.dataset_functions import slice_by_window
from classes.commons.utils import print_debug
import numpy as np
import math
import time
from python_speech_features import mfcc
from numpy import trapz
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from classes.converters.converter import Converter
from scipy.signal import argrelextrema


# dataset = set with all samples of only one activite
# item = new sample to classification

# Prepare data to DTW
def prepare_to_fastdtw(list):
    l_return = np.zeros(shape=(0, 2), dtype=np.int16)
    i = 0
    for l in list:
        aux = np.array([[i, l]], dtype=np.int16)
        l_return = np.append(l_return,aux,0)
        i = i + 1
    return l_return


# list = list of objects to slice
def slice_to_training_test(dataset, training_proportion=0.8, seed=1):

    print_debug("Calculating the size of lists...")
    list_len = len(dataset)
    training_len = int((list_len*training_proportion))
    test_len = list_len - training_len

    #Initialzing the list of test and training

    print_debug("Initializing the sets of training and test...")
    training_list = dataset.copy()
    test_list = []

    #List with test indexes to put in test list
    print_debug("Calculating the random indexes...")
    random.seed(seed)
    test_index = random.sample(range(training_len), test_len)

    print_debug("Loop for create the sets of training e test...")
    for index in test_index:
        test_list.append(dataset[index])
        training_list.pop(index)
    # Adding sequence time in dataframes
    times = list(range(50))
    #for d in training_list:
    #    d["time_s"] = list(range(0, len(d)))
    for d in test_list:
        for i in times:
            t = i


    print_debug("List len: {}".format(len(dataset)))
    print_debug("Training len: {}".format(len(training_list)))
    print_debug("Test len: {}".format(len(test_list)))
    print_debug("Training + test len: {}".format(len(training_list)+len(test_list)))

    return training_list, test_list


def get_dtw_mean(dataset , item):
    dtw_list = []
    for data in dataset:
        dtw, _ = fastdtw(data, item, dist=euclidean)
        dtw_list.append(dtw)
    return sum(dtw_list)/len(dtw_list)


def normalization(dataset):
    dataset_n = []
    mx = max(dataset)
    mn = min(dataset)
    for i in dataset:
        dataset_n.append(round(((i-mn)/(mx-mn)), 3))
    return dataset_n


def create_array_features(features_data_list):
    features = np.zeros(shape=(len(features_data_list[0]), len(features_data_list)))
    for index, features_data in enumerate(features_data_list):
        features[:, index] = features_data
    return features

def get_rms(dataset):
    x_2 = np.power(dataset, 2)
    return math.sqrt(np.sum(x_2)/len(dataset))


def get_mean(dataset):
    return np.mean(dataset)


def get_std(dataset):
    return np.std(dataset)


def get_minmax(dataset):
    return max(dataset) - min(dataset)


def get_integration(dataset):
    return trapz(dataset)


def get_kurtosis(dataset):
    return kurtosis(dataset)


def get_skew(dataset):
    return skew(dataset)


def get_correlation(var1, var2):
    return pearsonr(var1, var2)[0]

def calculating_features(dataframe_list):
    label = "activity"
    label_list = []

    # INTEGRATION #
    x_integration = []
    y_integration = []
    z_integration = []

    # RMS #
    x_rms = []
    y_rms = []
    z_rms = []

    # MINMAX #
    x_minmax = []
    y_minmax = []
    z_minmax = []

    # MEAN #
    x_mean = []
    y_mean = []
    z_mean = []

    # STANDARD DESVIATION #
    x_std = []
    y_std = []
    z_std = []

    # KURTOSIS #
    x_kurtosis = []
    y_kurtosis = []
    z_kurtosis = []

    # SKWESS #
    x_skew = []
    y_skew = []
    z_skew = []

    # CORRELATION #
    x_y = []
    x_z = []
    y_z = []

    for d in dataframe_list:
        x = d.loc[:, "x"]
        y = d.loc[:, "y"]
        z = d.loc[:, "z"]
        # INTEGRATION #
        ax = round(get_integration(normalization(x)), 3)
        ay = round(get_integration(normalization(y)), 3)
        az = round(get_integration(normalization(z)), 3)
        #i = round(get_integration(d.loc[:, "x"]), 3)
        x_integration.append(ax)
        y_integration.append(ay)
        z_integration.append(az)

        # RMS #
        x_r = round(get_rms(x),3)
        y_r = round(get_rms(x), 3)
        z_r = round(get_rms(x), 3)

        x_rms.append(x_r)
        y_rms.append(y_r)
        z_rms.append(z_r)

        # MINMAX #
        x_mm = round(get_minmax(x), 3)
        y_mm = round(get_minmax(y), 3)
        z_mm = round(get_minmax(z), 3)

        x_minmax.append(x_mm)
        y_minmax.append(y_mm)
        z_minmax.append(z_mm)

        # MEAN #
        x_m = round(get_mean(x), 3)
        y_m = round(get_mean(y), 3)
        z_m = round(get_mean(z), 3)

        x_mean.append(x_m)
        y_mean.append(y_m)
        z_mean.append(z_m)

        # STANDARD DESVIATION #
        x_sd = round(get_std(x), 3)
        y_sd = round(get_std(y), 3)
        z_sd = round(get_std(z), 3)

        x_std.append(x_sd)
        y_std.append(y_sd)
        z_std.append(z_sd)

        # KURTOSIS #
        x_k = round(get_kurtosis(x), 3)
        y_k = round(get_kurtosis(y), 3)
        z_k = round(get_kurtosis(z), 3)

        x_kurtosis.append(x_k)
        y_kurtosis.append(y_k)
        z_kurtosis.append(z_k)

        # SKWESS #
        x_sk = round(get_skew(x), 3)
        y_sk = round(get_skew(y), 3)
        z_sk = round(get_skew(z), 3)

        x_skew.append(x_sk)
        y_skew.append(y_sk)
        z_skew.append(z_sk)

        # CORRELATION #
        x_y.append(round(get_correlation(x, y), 3))
        x_z.append(round(get_correlation(x, z), 3))
        y_z.append(round(get_correlation(y, z), 3))

        # GET LABEL #
        label = d["activity"].iloc[0]
        label_list.append(label) #Get label for d


    #Initializing features array
    features = create_array_features([x_integration, y_integration, z_integration, x_rms, y_rms, z_rms,
                                      x_minmax, y_minmax, z_minmax, x_mean, y_mean, z_mean,
                                      x_std, y_std, z_std, x_kurtosis, y_kurtosis, z_kurtosis,
                                      x_y, x_z, y_z])
    print("Features Shape: {}".format(features.shape))

    #Initializing labels array
    labels = np.array(label_list)
    #labels = np.reshape(labels, (labels.shape[0],n1
    # 1))
    return features, labels


def load_training_data_with_window(dataset, filename, tablename, features, label,window_len=50):
    list_raw_data = dataset.load_list_of_activities(filename, tablename, features, False)
    list_window = slice_by_window(list_raw_data, label, window_len)
    training, test = slice_to_training_test(list_window)
    return training, test