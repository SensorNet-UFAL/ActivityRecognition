#===== Classification Tools =====#
from classes.commons.classification_functions import load_train_test_outlier_for_each_person
from classes.converters.arcma import ARCMAConverter
#===============================================================#

#===== Machine Learn =====#
from sklearn import svm
#===============================================================#

#===== Utils =====#
import numpy as np
#===============================================================#

#==== FUNCTIONS ====#

def get_sets_to_outlier_test(list_train_labels, list_test_labels, activity):

    train_indexes = np.where(list_train_labels != activity)
    test_indexes = np.where(list_test_labels != activity)
    outliers_indexes = np.where(list_train_labels == activity)

    return train_indexes, test_indexes, outliers_indexes

#Getting total accuracy for a features set.
def total_accuracy_for_set(features, list_train_features, list_train_labels, list_test_features, list_test_labels, activity_list):

    train_accuracy = 0
    test_accuracy = 0
    outliers_accuracy = 0

    count_loop = 0

    for i in activity_list:
        for p in range(0, len(list_train_features)):

            train_indexes, test_indexes, outliers_indexes = get_sets_to_outlier_test(list_train_labels[p], list_test_labels[p], i)

            train = list_train_features[p][train_indexes][:, features]
            test = list_test_features[p][test_indexes][:, features]
            outliers = list_train_features[p][outliers_indexes][:, features]
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            clf.fit(train)

            if len(train) > 0 and len(test) > 0 and len(outliers) > 0:
                count_loop = count_loop + 1;
                # predict
                pred_train = clf.predict(train)
                pred_test = clf.predict(test)
                pred_outliers = clf.predict(outliers)

                # errors
                n_error_train = pred_train[pred_train == -1].size
                n_error_test = pred_test[pred_test == -1].size
                n_error_outliers = pred_outliers[pred_outliers == 1].size

                #update accuracy
                train_accuracy = train_accuracy + (100 - (100 * (n_error_train / pred_train.size)))
                test_accuracy = test_accuracy + (100 - (100 * (n_error_test / pred_test.size)))
                outliers_accuracy = outliers_accuracy + (100 - (100 * (n_error_outliers / pred_outliers.size)))
    '''print("========= ARCMA ===========")
    print("Train Accuracy Mean = {}%".format(train_accuracy/ count_loop))
    print("Test Accuracy Mean = {}%".format(test_accuracy / count_loop))
    print("Outliers Accuracy Mean = {}%".format(outliers_accuracy / count_loop))'''
    return train_accuracy/count_loop, test_accuracy/count_loop, outliers_accuracy/count_loop


#Finding the best combination for 3 features
def find_the_best_set_depth_3_activity_loop(activity_list, dataset, filename, tablename, features, label_axis, label_column, window_len, person_indexes, person_column, additional_where = ""):

    list_train_features, list_train_labels, list_test_features, list_test_labels = load_train_test_outlier_for_each_person(dataset=dataset, filename=filename, tablename=tablename, features=features, label_axis=label_axis, label_column=label_column, window_len=window_len, person_indexes=person_indexes, person_column=person_column, additional_where=additional_where)
    print("##########################")
    print("###### TOTAL OF PERSONS: {} ######".format(len(list_train_features)))
    print("##########################")
    #Verificar resultado do outlier para cada trio possível de feature
    train_accuracy_flag = 0
    test_accuracy_flag = 0
    outliers_accuracy_flag = 0
    features_flag = []
    n_columns = list_train_features[0].shape[1]
    for c1 in range(n_columns-1):
        print("##########################")
        print("###### LOOP = {}/{} ######".format(c1, n_columns-2))
        print("##########################")
        for c2 in range(c1+1, n_columns):
            for c3 in range(c2 + 1, n_columns):
                train_accuracy, test_accuracy, outliers_accuracy = total_accuracy_for_set((c1, c2, c3), list_train_features, list_train_labels, list_test_features, list_test_labels, activity_list=activity_list)
                if(train_accuracy > 60 and test_accuracy > 60 and outliers_accuracy > 60):
                    if(train_accuracy > train_accuracy_flag and test_accuracy > test_accuracy_flag and outliers_accuracy > outliers_accuracy_flag):
                        train_accuracy_flag = train_accuracy
                        test_accuracy_flag = test_accuracy
                        outliers_accuracy_flag = outliers_accuracy
                        features_flag = [c1, c2, c3]
                        #print
                        print("----------------------------------------------------------")
                        print("Find Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], train_accuracy_flag))
                        print("Find Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], test_accuracy_flag))
                        print("Find Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], outliers_accuracy_flag))


    #print
    print("----------------------------------------------------------")
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], train_accuracy_flag))
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], test_accuracy_flag))
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], outliers_accuracy_flag))

##= ARCMA = ##
arcma = ARCMAConverter("..\\..\\\\databases\\arcma")
find_the_best_set_depth_3_activity_loop(activity_list=[1, 4, 7], dataset=arcma, filename="..\\arcma.db", tablename="arcma", features="x, y, z, activity", label_axis=["x","y","z"], label_column="activity", window_len=100, person_indexes=list(range(1,16)), person_column="person")
