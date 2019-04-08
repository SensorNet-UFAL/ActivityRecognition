###### IMPORTS ######
import os
from sklearn import svm
import sklearn
from sklearn.ensemble import IsolationForest
import numpy as np
from classes.converters.umaAdl import UmaAdlConverter
from classes.converters.arcma import ARCMAConverter
from classes.commons.classification_functions import calculating_features, load_training_data_with_window_from_sql
from sklearn.decomposition import PCA
from scripts.plots.view_classes import plot_with_two_dim
#################################################################

#LOADING ALL DATA AND CREATING ONE SET TO EACH PERSON WITH UMAFALL DATASET
def load_train_test_outlier_for_each_person_umafall():
    umafall = UmaAdlConverter('')
    list_train_features = []
    list_train_labels = []
    list_test_features = []
    list_test_labels = []
    for p in range(1, 3):
        training, test = load_training_data_with_window_from_sql(umafall, "..\\..\\..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity from umafall where SensorType=0 and SensorID=1 and person={} and (activity='Hopping' or activity='Bending' or activity='Sitting')".format(p), "activity", 50)
        training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        list_train_features.append(training_features)
        list_train_labels.append(training_labels)
        list_test_features.append(test_features)
        list_test_labels.append(test_labels)
    return list_train_features, list_train_labels, list_test_features, list_test_labels

#LOADING ALL DATA AND CREATING ONE SET TO EACH PERSON WITH ARCMA DATASET
def load_train_test_outlier_for_each_person_arcma():
    arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
    list_train_features = []
    list_train_labels = []
    list_test_features = []
    list_test_labels = []
    for p in range(1, 4):
        training, test = load_training_data_with_window_from_sql(arcma, "..\\..\\..\\arcma.db", "Select x, y, z, activity from arcma where person={} and (activity <> 2 and activity <> 3)".format(p), "activity", 50)
        training_features, training_labels = calculating_features(training, x_label="x", y_label="y", z_label="z")
        test_features, test_labels = calculating_features(test, x_label="x", y_label="y", z_label="z")
        list_train_features.append(training_features)
        list_train_labels.append(training_labels)
        list_test_features.append(test_features)
        list_test_labels.append(test_labels)
    return list_train_features, list_train_labels, list_test_features, list_test_labels

def get_sets_to_outlier_test(list_train_labels, list_test_labels, activity):

    train_indexes = np.where(list_train_labels != activity)
    test_indexes = np.where(list_test_labels != activity)
    outliers_indexes = np.where(list_train_labels == activity)

    return train_indexes, test_indexes, outliers_indexes

#FINDING MEAN ACCURACY TO THE DATASET: TRAIN, TEST AND OUTLIER
def accurary_mean_discard_one_activity_by_loop(activity_list):

    #list_train_features, list_train_labels, list_test_features, list_test_labels = load_train_test_outlier_for_each_person_umafall()
    list_train_features, list_train_labels, list_test_features, list_test_labels = load_train_test_outlier_for_each_person_arcma()
    somafor = 0

    for activity in activity_list:

        train_accuracy = 0
        test_accuracy = 0
        outliers_accuracy = 0
        count_loop = 0

        for person in range(0, len(list_train_features)):

            somafor = somafor + 1

            #list_test_labels[person][list_test_labels[person] == 'backwardFall'] = 'fall'
            #list_test_labels[person][list_test_labels[person] == 'forwardFall'] = 'fall'
            #list_test_labels[person][list_test_labels[person] == 'lateralFall'] = 'fall'

            train_indexes, test_indexes, outliers_indexes = get_sets_to_outlier_test(list_train_labels[person],
                                                                                     list_test_labels[person], activity)


            train = list_train_features[person][train_indexes]
            test = list_test_features[person][test_indexes]
            outliers = list_train_features[person][outliers_indexes]

            if len(train) > 0 and len(test) > 0 and len(outliers) > 0:
                train_model_pca = PCA(n_components=5).fit(train)
                test_model_pca = PCA(n_components=5).fit(test)
                outlier_model_pca = PCA(n_components=5).fit(outliers)

                train_pca = train_model_pca.transform(train)
                test_pca = test_model_pca.transform(test)
                outlier_pca = outlier_model_pca.transform(outliers)

                #train_pca = train[:,(11,13,15)]
                #test_pca = test[:,(11,13,15)]
                #outlier_pca = outliers[:,(11,13,15)]


                clf = svm.OneClassSVM(nu=0.3, kernel="rbf", gamma='scale')
                #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                #rng = np.random.RandomState(10)
                #clf = IsolationForest(behaviour='new', max_samples=1000, random_state=rng, contamination=0.1)
                clf.fit(train_pca)

                # predict
                #verifica shape
                #print("Shape Train: {}".format(train_pca.shape))
                #print("Shape Test: {}".format(test_pca.shape))
                #print("Shape Outlier: {}".format(outlier_pca.shape))
                pred_train = clf.predict(train_pca)
                pred_test = clf.predict(test_pca)
                pred_outliers = clf.predict(outlier_pca)

                # errors
                n_error_train = pred_train[pred_train == -1].size
                n_error_test = pred_test[pred_test == -1].size
                n_error_outliers = pred_outliers[pred_outliers == 1].size

                # update accuracy
                train_accuracy = train_accuracy + (100 - (100 * (n_error_train / pred_train.size)))
                test_accuracy = test_accuracy + (100 - (100 * (n_error_test / pred_test.size)))
                outliers_accuracy = outliers_accuracy + (100 - (100 * (n_error_outliers / pred_outliers.size)))

                count_loop = count_loop + 1;
        if count_loop > 0:
            print("========= ACCURACY MEAN ===========")
            print("ATIVIDADE OUTLIER: {}".format(activity))
            print("Train Accuracy Mean = {}%".format(train_accuracy/count_loop))
            print("Test Accuracy Mean = {}%".format(test_accuracy/count_loop))
            print("Outliers Accuracy Mean = {}%".format(outliers_accuracy/count_loop))
    print("Times of loop: {}".format(somafor))
    print("Len person: {}".format(len(list_train_features)))


#accurary_mean_discard_one_activity_by_loop(activity_list=['Hopping', 'Bending', 'Sitting'])
accurary_mean_discard_one_activity_by_loop(activity_list=[1,4,5,6,7])
#TODO
#Retirar as atividades na consulta SQL, da forma que se encontra, está removendo apenas do loop para formar o conjunto de outlier.