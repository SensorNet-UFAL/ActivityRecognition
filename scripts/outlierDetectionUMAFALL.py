import os
from classes.commons.classification_functions import calculating_features, load_training_data_with_window_from_sql
from classes.converters.arcma import ARCMAConverter
from sklearn import svm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from classes.converters.umaAdl import UmaAdlConverter
import time

umafall = UmaAdlConverter('')

def load_all_data_to_outlier_test_umafall():
    list_train_features = []
    list_train_labels = []
    list_test_features = []
    list_test_labels = []
    for p in range(1, 18):
        training, test = load_training_data_with_window_from_sql(umafall, "..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity, person from umafall where SensorType=0 and SensorID=0 and person={}".format(p), "activity", 50)
        training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        list_train_features.append(training_features)
        list_train_labels.append(training_labels)
        list_test_features.append(test_features)
        list_test_labels.append(test_labels)
    return list_train_features, list_train_labels, list_test_features, list_test_labels

def load_data_to_outlier_test_umafall(activity_outlier, person_list = 16):

    #load data
    training, test = load_training_data_with_window_from_sql(umafall, "..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity from arcma where SensorType=0 and SensorID=0 and activity <> {} and person={}".format(activity_outlier, person_list), "activity", 50)
    training_features, training_labels = calculating_features(training)
    test_features, test_labels = calculating_features(test)
    _, test_outliers = load_training_data_with_window_from_sql(umafall, "..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity from arcma where SensorType=0 and SensorID=0 and activity = {} and person={}".format(activity_outlier, person_list), "activity", 50)
    outliers_test_features, _ = calculating_features(test_outliers)

    #consultar subconjuntos no dataframe pandas => https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas

    return training_features, test_features, outliers_test_features

def load_data_to_outlier_test_umafall(activity_outlier):

    #load data
    training, test = load_training_data_with_window_from_sql(umafall, "..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity from umafall where activity <> '{}' and person={}".format(activity_outlier, 1), "activity", 50)
    training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
    test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
    _, test_outliers = load_training_data_with_window_from_sql(umafall, "..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity from umafall where activity = '{}' and person={}".format(activity_outlier, 1), "activity", 50)
    outliers_test_features, _ = calculating_features(test_outliers,  x_label="XAxis", y_label="YAxis", z_label="ZAxis")

    return training_features, test_features, outliers_test_features

#DEPTH 2
def find_the_best_set_depth_2():

    training_features, test_features, outliers_test_features = load_data_to_outlier_test_umafall(7)

    #Verificar resultado do outlier para cada dupla possível de feature
    n_columns = training_features.shape[1]
    for c in range(n_columns-1):
        for c2 in range(c+1, n_columns):
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            clf.fit(training_features[:, (c, c2)])

            #predict
            pred_train = clf.predict(training_features[:, (c, c2)])
            pred_test = clf.predict(test_features[:, (c, c2)])
            pred_outliers = clf.predict(outliers_test_features[:, (c, c2)])

            #errors
            n_error_train = pred_train[pred_train == -1].size
            n_error_test = pred_test[pred_test == -1].size
            n_error_outliers = pred_outliers[pred_outliers == 1].size

            #prints
            print("----------------------------------------------------------")
            print("Train Accuracy - columns [{}], [{}]: {}/{} - {}%".format(c, c2, n_error_train, pred_train.size, 100 - (100 * (n_error_train / pred_train.size))))
            print("Test Accuracy - columns [{}], [{}]: {}/{} - {}%".format(c, c2, n_error_test, pred_test.size, 100 - (100 * (n_error_test / pred_test.size))))
            print("Outliers Accuracy - columns [{}], [{}]: {}/{} - {}%".format(c, c2, n_error_outliers, pred_outliers.size,
                                                       100 - (100 * (n_error_outliers / pred_outliers.size))))

#DEPTH 3
def find_the_best_set_depth_3():

    training_features, test_features, outliers_test_features = load_data_to_outlier_test_umafall(7)

    #Verificar resultado do outlier para cada dupla possível de feature
    train_accuracy_flag = 0
    test_accuracy_flag = 0
    outliers_accuracy_flag = 0
    features_flag = []
    n_columns = training_features.shape[1]
    for c1 in range(n_columns-1):
        for c2 in range(c1+1, n_columns):
            for c3 in range(c2 + 1, n_columns):
                train = training_features[:, (c1, c2, c3)]
                test = test_features[:, (c1, c2, c3)]
                outliers = outliers_test_features[:, (c1, c2, c3)]

                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                clf.fit(train)

                # predict
                pred_train = clf.predict(train)
                pred_test = clf.predict(test)
                pred_outliers = clf.predict(outliers)

                # errors
                n_error_train = pred_train[pred_train == -1].size
                n_error_test = pred_test[pred_test == -1].size
                n_error_outliers = pred_outliers[pred_outliers == 1].size

                # update accuracy
                train_accuracy = (100 - (100 * (n_error_train / pred_train.size)))
                test_accuracy = (100 - (100 * (n_error_test / pred_test.size)))
                outliers_accuracy =  (100 - (100 * (n_error_outliers / pred_outliers.size)))
                if(train_accuracy > 70 and test_accuracy > 70 and outliers_accuracy > 70):
                    if(train_accuracy > train_accuracy_flag and test_accuracy > test_accuracy_flag and outliers_accuracy > outliers_accuracy_flag):

                        train_accuracy_flag = train_accuracy
                        test_accuracy_flag = test_accuracy
                        outliers_accuracy_flag = outliers_accuracy
                        features_flag = [c1, c2, c3]

    #print
    print("----------------------------------------------------------")
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], train_accuracy_flag))
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], test_accuracy_flag))
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], outliers_accuracy_flag))

#DEPTH 3 WITH ACTIVITY LOOP
def find_the_best_set_depth_3_activity_loop(activity_list):

    list_train_features, list_train_labels, list_test_features, list_test_labels = load_all_data_to_outlier_test_umafall()
    print("##########################")
    print("###### TOTAL OF PERSONS: {} ######".format(len(list_train_features)))
    print("##########################")
    #Verificar resultado do outlier para cada dupla possível de feature
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
                if(train_accuracy > 70 and test_accuracy > 70 and outliers_accuracy > 70):
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

def get_sets_to_outlier_test(list_train_features, list_train_labels, list_test_features, list_test_labels, activity):

    train_indexes = np.where(list_train_labels != activity)
    test_indexes = np.where(list_test_labels != activity)
    outliers_indexes = np.where(list_train_labels == activity)

    return train_indexes, test_indexes, outliers_indexes

def total_accuracy_for_set(features, list_train_features, list_train_labels, list_test_features, list_test_labels, activity_list):

    # ARCMA #
    train_accuracy = 0
    test_accuracy = 0
    outliers_accuracy = 0

    count_loop = 0

    for i in activity_list:
        for p in range(0, len(list_train_features)):

            train_indexes, test_indexes, outliers_indexes = get_sets_to_outlier_test(list_train_features[p], list_train_labels[p], list_test_features[p], list_test_labels[p], i)

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


def plot_partial_set(test_features, test_labels):
    labels = np.unique(test_labels)
    colors = ["pink", "black", "green", "red", "blue", "purple", "yellow"]
    plt.scatter(test_features[:, 0], test_features[:, 3], c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.colorbar()
    plt.show()

#find_the_best_set_depth_2()
#find_the_best_set_depth_3()
find_the_best_set_depth_3_activity_loop(activity_list=["Bending","forwardFall","Hopping","Jogging","LyingDown","Sitting","Walking","backwardFall","lateralFall","GoDownstairs","GoUpstairs"])
#load_all_data_to_outlier_test_arcma()

#Testar acuária para 1 conjunto de features
#list_train_features, list_train_labels, list_test_features, list_test_labels = load_all_data_to_outlier_test_umafall()
#total_accuracy_for_set((11, 13,15), list_train_features, list_train_labels, list_test_features, list_test_labels, activity_list=["Bending","forwardFall","Hopping","Jogging","LyingDown","Sitting","Walking","backwardFall","lateralFall","GoDownstairs","GoUpstairs"])


