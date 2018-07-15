import random
from classes.commons.utils import print_debug

#dataframe = object Pandas DataFrame
def slice_by_window(dataframe, window_length):
    index = 0
    dataframe_len = len(dataframe)
    result = []
    while(index < dataframe_len):
        try:
            result.append(dataframe.iloc[index:(index+window_length)])
            index = index + window_length
        except Exception:
            break
    return result