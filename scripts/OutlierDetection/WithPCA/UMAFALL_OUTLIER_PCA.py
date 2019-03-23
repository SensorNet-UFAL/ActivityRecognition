###### IMPORTS ######
from sklearn import svm
import numpy as np
from classes.converters.umaAdl import UmaAdlConverter
from classes.commons.classification_functions import calculating_features, load_training_data_with_window_from_sql
from sklearn.decomposition import PCA
from scripts.plots.view_classes import plot_with_two_dim
#################################################################

#LOADING ALL DATA AND CREATING ONE SET TO EACH PERSON
def load_train_test_outlier_for_each_person():
    umafall = UmaAdlConverter('')
    list_train_features = []
    list_train_labels = []
    list_test_features = []
    list_test_labels = []
    for p in range(1, 3):
        training, test = load_training_data_with_window_from_sql(umafall, "..\\..\\..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity, person from umafall where SensorType=0 and SensorID=0 and person={}".format(p), "activity", 50)
        training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
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

    list_train_features, list_train_labels, list_test_features, list_test_labels = load_train_test_outlier_for_each_person()
    somafor = 0

    for activity in activity_list:

        train_accuracy = 0
        test_accuracy = 0
        outliers_accuracy = 0
        count_loop = 0

        for person in range(0, len(list_train_features)):

            somafor = somafor + 1

            train_indexes, test_indexes, outliers_indexes = get_sets_to_outlier_test(list_train_labels[person],
                                                                                     list_test_labels[person], activity)

            train = list_train_features[person][train_indexes]
            test = list_test_features[person][test_indexes]
            outliers = list_train_features[person][outliers_indexes]

            if len(train) > 0 and len(test) > 0 and len(outliers) > 0:

                train_pca = PCA(n_components=2).fit_transform(train)
                test_pca = PCA(n_components=2).fit_transform(test)
                outlier_pca = PCA(n_components=2).fit_transform(outliers)

                l1 = np.zeros(train_pca.shape[0])
                l2 = np.ones(outlier_pca.shape[0])

                data_to_plot_pca = np.concatenate((train_pca, outlier_pca))
                data_to_plot = np.concatenate((train, outliers))
                labels = np.concatenate((l1,l2))

                clf = svm.OneClassSVM(nu=0.05, gamma=0.1)
                clf = clf.fit(train)
                plot_with_two_dim(clf, data_to_plot, labels)

                clf = svm.OneClassSVM(nu=0.05, gamma=0.1)
                clf = clf.fit(train_pca)
                plot_with_two_dim(clf, data_to_plot_pca, labels)

                '''
                #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                clf = svm.OneClassSVM(nu=0.05, gamma=0.1)
                clf.fit(train_pca)

                # predict
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

                count_loop = count_loop + 1;'''
        if count_loop > 0:
            print("========= ACCURACY MEAN ===========")
            print("Train Accuracy Mean = {}%".format(train_accuracy/count_loop))
            print("Test Accuracy Mean = {}%".format(test_accuracy/count_loop))
            print("Outliers Accuracy Mean = {}%".format(outliers_accuracy/count_loop))
    print("Times of loop: {}".format(somafor))
    print("Len person: {}".format(len(list_train_features)))


accurary_mean_discard_one_activity_by_loop(activity_list=["Bending","forwardFall","Hopping","Jogging","LyingDown","Sitting","Walking","backwardFall","lateralFall","GoDownstairs","GoUpstairs"])