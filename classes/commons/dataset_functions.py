import random
from classes.commons.utils import print_debug
import traceback

# dataframe = object Pandas DataFrame
def slice_by_window(dataframe, label, window_length):
    index = 0
    dataframe_len = len(dataframe)
    result = []
    while(index < dataframe_len):
        try:
            l = dataframe.iloc[index:(index+window_length)]
            #l = unique_label(l, label)
            result.append(l)
            index = index + window_length
        except Exception as e:
            print_debug(e)
            print(traceback.format_exc())
            break
    return result


def unique_label(dataframe, label):
    l = list(dataframe.loc[:, label])
    set_list = set(l)
    if len(set_list) > 1:
        print_debug("Find more that one label in the window.")
        item_count = 0
        u_label = None
        for i in set_list:
            if item_count < l.count(i):
                item_count = l.count(i)
                u_label = i
        dataframe.loc[:, label] = u_label
        #print_debug(dataframe.loc[:, label])
    return dataframe
