from classes.converters.owndata import OwnDataConverter
from classes.commons.classification_functions import load_training_data_with_window_from_person, load_training_data_with_window_from_sql, calculating_features
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import numpy as np
import os

debug = True
directory = "{}\\databases\\owndata".format(os.getcwd())
owndata = OwnDataConverter(directory)
def classification():
    training, test = load_training_data_with_window_from_person(owndata, directory+"\\ActivityRecords.sqlite", "ActivityRecords", "x, y, z, activity_tag",
                                                    "activity_tag", window_len=25, person_tag = 1, person_column = 'person_tag')
    training_features, training_labels = calculating_features(training, "activity_tag")
    test_features, test_labels = calculating_features(test, "activity_tag")

    extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
    extra_trees.fit(training_features, training_labels)
    print("Accuracy Extra Trees: {}".format(accuracy_score(test_labels, extra_trees.predict(test_features))))


def detect_outliers():
    training, test = load_training_data_with_window_from_sql(owndata, directory+"\\ActivityRecords.sqlite",
                                                             "select x, y, z, activity_tag from ActivityRecords where person_tag = 1 and (activity_tag = 1 or activity_tag = 11 or activity_tag = 5 or activity_tag = 6 or activity_tag = 14)",
                                                             "activity_tag", 25)
    training_features, training_labels = calculating_features(training, "activity_tag")
    test_features, test_labels = calculating_features(test, "activity_tag")

    #-----------TESTE--------------#

    training2, test2 = load_training_data_with_window_from_sql(owndata, directory + "\\ActivityRecords.sqlite",
                                                             "select x, y, z, activity_tag from ActivityRecords where person_tag = 1 and (activity_tag = 15 or activity_tag = 19)",
                                                             "activity_tag", 25)
    training_features2, training_labels2 = calculating_features(training2, "activity_tag")

    #--------------------------------

    outliers_fraction = 0.05
    rng = np.random.RandomState(42)
    #clf = IsolationForest(max_samples=200, contamination=0.15, random_state=rng) # 84% - Detecta verdadeiras ocorrencias e 71% - Detecta outliers - outliers_fraction = 0.05
    clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.01) # 90% - Detecta verdadeiras ocorrencias e 85% - Detecta outliers
    #clf = LocalOutlierFactor( n_neighbors=35, contamination=outliers_fraction) # MUITO RUIN
    #clf = EllipticEnvelope(contamination=outliers_fraction) # 89% - Detecta verdadeiras ocorrências e 71% - Detecta outliers
    clf.fit(training_features)
    y_pred_train = clf.predict(training_features)
    print(y_pred_train)
    print((y_pred_train.size - np.count_nonzero(y_pred_train == -1))/y_pred_train.size)


classification()
#detect_outliers()