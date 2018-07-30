#IMPORTS#
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from classes.commons.dataset_functions import slice_by_window
from classes.commons.utils import print_debug
import numpy as np
import time
from python_speech_features import mfcc
from numpy import trapz
import matplotlib.pyplot as plt
from classes.converters.converter import Converter
from sklearn.neighbors import KNeighborsClassifier

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

def get_integration(dataset):
    return trapz(dataset)

def calculating_features(dataframe_list):
    label = "activity"
    label_list = []
    #Features for X
    x_integration = []
    for d in dataframe_list:
        label = d["activity"].iloc[0]
        i = round(get_integration(normalization(d.loc[:,"x"])), 3)
        x_integration.append(i)
        label_list.append(label) #Get label for d


    #====Training kNN====#
    #Initializing features array
    n_features = 1 # Number of using features
    features = np.zeros(shape=(len(x_integration),n_features))
    features[:,0] = x_integration

    #Initializing labels array
    labels = np.array(label_list)

    #Fit kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)


def load_training_data_with_window(dataset, filename, tablename, features, label,window_len=50):
    list_raw_data = dataset.load_list_of_activities(filename, tablename, features, False)
    list_window = slice_by_window(list_raw_data, label, window_len)
    training, test = slice_to_training_test(list_window)
    calculating_features(training)
    #ta = time.time()
    #tr = prepare_to_fastdtw(list(training[0].loc[:, "x"]))
    #print("MFCC SHAPE: {}".format(mfcc_feature.shape))
    ''' for t in training:
        tr = prepare_to_fastdtw(list(t.loc[:,"x"]))
        ts = prepare_to_fastdtw(list(test[0].loc[:,"x"]))
        dtw_mean = get_dtw_mean(tr, ts)
    print("Seconds: {}s".format(time.time()-ta))
    
    PLOT
    t = range(0,len(training[0].loc[:,"x"]))
    x = training[10].loc[:,"x"]
    plt.plot(t,x)
    plt.xlabel('tempo (ms)')
    plt.ylabel('Acelerômetro - eixo X')
    plt.title('Atividade para Classificar')
    plt.show()'''