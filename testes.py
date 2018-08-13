import os
from classes.converters.arcma import ARCMAConverter
from classes.commons.classification_functions import load_training_data_with_window, calculating_features
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
debug = True
import time
import traceback
from fastdtw import fastdtw
import numpy as np

def filter_by_activity(dataset, label):
    r = []
    for i in dataset:
        try:
            #print(list(i.loc[:, "activity"]))
            if list(i.loc[:, "activity"])[0] == label:
                r.append(i)
        except Exception as e:
            print(i)
            traceback.print_exc()
            break
    return r

def get_dtw_mean(dataset , item):
    dtw_list = []
    for data in dataset:
        dtw, _ = fastdtw(data, item, dist=euclidean)
        dtw_list.append(dtw)
    return sum(dtw_list)/len(dtw_list)

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
training, test = load_training_data_with_window(arcma, "arcma.db", "arcma",
                                                "x, y, z, activity","activity", window_len=100)

activity1 = filter_by_activity(training,2)
x = list(range(0,100))
plt.plot(x, activity1[0].loc[:,"x"])
plt.plot(x, activity1[1].loc[:,"x"])
plt.plot(x, activity1[2].loc[:,"x"])
plt.show()
