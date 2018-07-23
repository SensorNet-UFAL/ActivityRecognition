#IMPORTS#
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from classes.commons.dataset_functions import slice_by_window
from classes.commons.utils import print_debug
from classes.converters.converter import Converter

# dataset = set with all samples of only one activite
# item = new sample to classification

#list  = list of objects to slice
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

def load_training_data_with_window(dataset, filename, tablename, features, label,window_len=50):
    list_raw_data = dataset.load_list_of_activities(filename, tablename, features, False)
    list_window = slice_by_window(list_raw_data, label, window_len)
    training, test = slice_to_training_test(list_window)
    print(test[0])
