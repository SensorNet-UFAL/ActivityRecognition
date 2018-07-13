
#list  = list of objects to slice
def slice_to_training_test(list, training_proportion):
    list_len = len(list)
    training_len = int((list_len*training_proportion))
    test_len = list_len - training_len

    training_list = list[0:training_len]
    test_list = list[training_len:list_len]
    #for to training = list
    '''for i in range(training_len,list_len):
        print("Pop: {}".format(i))
        print("List len 1: {}".format(list_len))
        print("List len A: {}".format(len(list)))
        test_list.append(training_list.pop(i))
        '''
    print("list len: {}".format(list_len))
    print("Training len: {}".format(len(training_list)))
    print("Test len: {}".format(len(test_list)))




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