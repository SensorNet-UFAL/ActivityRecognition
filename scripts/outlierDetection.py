import os
from classes.commons.classification_functions import calculating_features, load_training_data_with_window_from_sql
from classes.converters.arcma import ARCMAConverter
from sklearn import svm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))

#configuration
person_index = 7
activity_outlier = 1

#load data
training, test = load_training_data_with_window_from_sql(arcma, "..\\arcma.db", "Select x, y, z, activity from arcma where activity <> {} and person={}".format(activity_outlier, 13), "activity", 50)
training_features, training_labels = calculating_features(training)
test_features, test_labels = calculating_features(test)
full_data = np.concatenate((training_features, test_features))
_, test_outliers = load_training_data_with_window_from_sql(arcma, "..\\arcma.db", "Select x, y, z, activity from arcma where activity = {} and person={}".format(activity_outlier, 13), "activity", 50)
outliers_test_features, _ = calculating_features(test_outliers)

#fit
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(full_data)

#test
pred_full = clf.predict(training_features)
pred_train = clf.predict(training_features)
pred_test = clf.predict(test_features)
pred_outliers = clf.predict(outliers_test_features)

#error
n_error_full = pred_full[pred_full == -1].size
n_error_train = pred_train[pred_train == -1].size
n_error_test = pred_test[pred_test == -1].size
n_error_outliers = pred_outliers[pred_outliers == 1].size

#print
print("Error full: {}/{} - {}%".format(n_error_full, pred_full.size, 100*(n_error_full/pred_full.size)))
print("Error train: {}/{} - {}%".format(n_error_train, pred_train.size, 100*(n_error_train/pred_train.size)))
print("Error test: {}/{} - {}%".format(n_error_test, pred_test.size, 100*(n_error_test/pred_test.size)))
print("Error outliers: {}/{} - {}%".format(n_error_outliers, pred_outliers.size, 100*(n_error_outliers/pred_outliers.size)))

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
        print("Error train columns [{}], [{}]: {}/{} - {}%".format(c, c2, n_error_train, pred_train.size, 100 * (n_error_train / pred_train.size)))
        print("Error test columns [{}], [{}]: {}/{} - {}%".format(c, c2, n_error_test, pred_test.size, 100 * (n_error_test / pred_test.size)))
        print("Error outliers columns [{}], [{}]: {}/{} - {}%".format(c, c2, n_error_outliers, pred_outliers.size,
                                                   100 * (n_error_outliers / pred_outliers.size)))



def plot_partial_set():
    labels = np.unique(test_labels)
    colors = ["pink", "black", "green", "red", "blue", "purple", "yellow"]
    plt.scatter(test_features[:, 0], test_features[:, 3], c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.colorbar()
    plt.show()