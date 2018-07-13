#IMPORTS#
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# dataset = set with all samples of only one activite
# item = new sample to classification

def get_dtw_mean(dataset , item):
    dtw_list = []
    for data in dataset:
        dtw, _ = fastdtw(data, item, dist=euclidean)
        dtw_list.append(dtw)
    return sum(dtw_list)/len(dtw_list)