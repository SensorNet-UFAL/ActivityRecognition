
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm #SVM
from random import randint
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import numpy as np
from classes.converters.umaAdl import UmaAdlConverter
from classes.commons.classification_functions import load_training_data_with_window_from_sql, calculating_features
from classes.commons.utils import verify_confusion_matrix_simple
import time

most_accurate_person = {"clf": "", "id": 0, "accuracy": 0}

def put_clf_init_values(clf_store, clf, person_index, training_features, training_labels, test_features, test_labels):
    clf_store['clf'] = clf
    clf_store['training_features'] = training_features
    clf_store['training_labels'] = training_labels
    clf_store['test_features'] = test_features
    clf_store['test_labels'] = test_labels
    clf_store['person'] = person_index

def verify_fit(clf_store):
    start = time.time()
    clf = clf_store['clf']
    clf.fit(clf_store['training_features'], clf_store['training_labels'])
    elapsed = time.time() - start
    clf_store['time_fit'] += elapsed

def verify_accuracy(clf_store):
    start = time.time()
    clf = clf_store['clf']
    pred = clf.predict(clf_store['test_features'])
    accuracy = accuracy_score(clf_store['test_labels'], pred)
    clf_store['accuracy'] += accuracy
    clf_store['f1'] += f1_score(clf_store['test_labels'], pred, average='macro')
    elapsed = time.time() - start
    clf_store['time_classify'] += elapsed
    clf_store['len'] += 1

    #Most accurate person
    if accuracy > most_accurate_person["accuracy"] and clf_store['person'] != 14 and clf_store['person'] != 17:
        most_accurate_person['accuracy'] = accuracy
        most_accurate_person['id'] = clf_store['person']
        most_accurate_person['clf'] = clf_store['name']

def print_accuracy(clf_store):
    print("Time to predict: {}s".format(clf_store['time_classify']/clf_store['len']))
    print("Accuracy: {}".format(clf_store['accuracy'] / clf_store['len']))
    print("F-score: {}".format(clf_store['f1'] / clf_store['len']))
    print("Most Accuracy: CLF: {}, Person: {}, Accuracy: {}".format(most_accurate_person['clf'], most_accurate_person['id'], most_accurate_person['accuracy']))

def print_time_to_fit(clf_store):
    print("---------------------------------------")
    print("Classifier: {}".format(clf_store['name']))
    print("Time to fit: {}s".format(clf_store['time_fit'] / clf_store['len']))

def create_umafall_db():
    umafall = UmaAdlConverter('')
    umafall.load_from_csv("C:\\Users\\wylken.machado.INTRA\\ownCloud\\WYLKEN\\MESTRADO\\2016\\Dissertacao\\Implementacoes\\Projeto_Artigo\\Datasets\\UMA_ADL_FALL_Dataset")
    umafall.convert_csv_to_sql("umafall.db", "umafall")

def classification():
    umafall = UmaAdlConverter('')

    # --init classifier temp--#
    knn_store = {"name": "KNN", "person": "", "clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0,
                 "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    nb_store = {"name": "Naive Bayes", "person": "", "clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0,
                "time_classify": 0,
                "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    extra_trees_store = {"name": "Extra-Trees", "person": "", "clf": None, "len": 0, "accuracy": 0, "f1": 0,
                         "time_fit": 0,
                         "time_classify": 0,
                         "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    random_forest_store = {"name": "Random Forest", "person": "", "clf": None, "len": 0, "accuracy": 0, "f1": 0,
                           "time_fit": 0,
                           "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None,
                           "test_labels": None}
    decision_tree_store = {"name": "Decision Tree", "person": "", "clf": None, "len": 0, "accuracy": 0, "f1": 0,
                           "time_fit": 0,
                           "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None,
                           "test_labels": None}
    svm_store = {"name": "SVM", "person": "", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "f1": 0,
                 "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}

    for person_index in range(1, 18):
        print("==== Person {} ====".format(person_index))
        training, test = load_training_data_with_window_from_sql(umafall, "umafall.db", "Select XAxis, YAxis, ZAxis, activity from umafall where SensorType=0 and SensorID=0 and person={}".format(person_index), "activity", 50)
        training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        if len(training_labels) > 10:
            # ==KNN=#
            knn = KNeighborsClassifier(n_neighbors=5)
            put_clf_init_values(knn_store, knn, person_index, training_features, training_labels, test_features, test_labels)
            verify_fit(knn_store)
            verify_accuracy(knn_store)

            # ==Naive Bayes==#
            nb = GaussianNB()
            put_clf_init_values(nb_store, nb, person_index, training_features, training_labels, test_features, test_labels)
            verify_fit(nb_store)
            verify_accuracy(nb_store)

            # ==Extra-Trees==#
            extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
            put_clf_init_values(extra_trees_store, extra_trees, person_index, training_features, training_labels, test_features,
                                test_labels)
            verify_fit(extra_trees_store)
            verify_accuracy(extra_trees_store)

            # ==Random Forest==#
            random_forest = RandomForestClassifier(max_depth=20, random_state=1)
            put_clf_init_values(random_forest_store, random_forest, person_index, training_features, training_labels, test_features,
                                test_labels)
            verify_fit(random_forest_store)
            verify_accuracy(random_forest_store)

            # ==Decision Tree==#
            decision_tree = tree.DecisionTreeClassifier()
            put_clf_init_values(decision_tree_store, decision_tree, person_index, training_features, training_labels, test_features,
                                test_labels)
            verify_fit(decision_tree_store)
            verify_accuracy(decision_tree_store)

            # ==SVM=#
            svm_clf = svm.SVC()
            put_clf_init_values(svm_store, svm_clf, person_index, training_features, training_labels, test_features, test_labels)
            verify_fit(svm_store)
            verify_accuracy(svm_store)

            #== Confusion Matrix ==#
            #verify_confusion_matrix_simple(test_labels, pred, "Matriz de Confusão - Extre Trees.")

        else:
            print("DON'T HAVE DATA")
        print("==============================")

    print_time_to_fit(knn_store)
    print_accuracy(knn_store)

    print_time_to_fit(nb_store)
    print_accuracy(nb_store)

    print_time_to_fit(svm_store)
    print_accuracy(svm_store)

    print_time_to_fit(extra_trees_store)
    print_accuracy(extra_trees_store)

    print_time_to_fit(random_forest_store)
    print_accuracy(random_forest_store)

    print_time_to_fit(decision_tree_store)
    print_accuracy(decision_tree_store)

def get_confusion_matrix():

    umafall = UmaAdlConverter('')
    training, test = load_training_data_with_window_from_sql(umafall, "umafall.db",
                                                             "Select XAxis, YAxis, ZAxis, activity from umafall where SensorType=0 and SensorID=0 and person={}".format(
                                                                 13), "activity", 50)
    # Calculate Features
    training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
    test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")

    extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
    extra_trees.fit(training_features, training_labels)
    pred = extra_trees.predict(test_features)
    print("Accuracy Extra Trees: {}".format(accuracy_score(test_labels, pred)))
    verify_confusion_matrix_simple(test_labels, pred, "Matriz de Confusão - Extra-Trees.")

def verify_accuracy_with_noise():
    umafall = UmaAdlConverter('')
    training, test = load_training_data_with_window_from_sql(umafall, "umafall.db",
                                                             "Select XAxis, YAxis, ZAxis, activity from umafall where SensorType=0 and SensorID=0 and person={}".format(
                                                                 13), "activity", 50)
    training_i, test_i = load_training_data_with_window_from_sql(umafall, "umafall.db",
                                                             "Select XAxis, YAxis, ZAxis, activity from umafall where SensorType=0 and SensorID=0 and person={} and activity != '{}'".format(
                                                                 13,'Sitting'), "activity", 50)

    training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis",
                                                              z_label="ZAxis")
    test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis",
                                                                  z_label="ZAxis")

    training_features_i, training_labels_i = calculating_features(training_i, x_label="XAxis", y_label="YAxis",
                                                              z_label="ZAxis")
    test_features_i, test_labels_i = calculating_features(test_i, x_label="XAxis", y_label="YAxis", z_label="ZAxis")

    extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
    extra_trees.fit(training_features_i, training_labels_i)
    pred = extra_trees.predict(test_features_i)
    pred_noise = extra_trees.predict(test_features)
    print("Accuracy Extra Trees: {}".format(accuracy_score(test_labels_i, pred)))
    print("Accuracy with Noise Extra Trees: {}".format(accuracy_score(test_labels, pred_noise)))
    #verify_confusion_matrix_simple(test_labels, pred_noise, "Matriz de Confusão - Extra-Trees.")

    # CONFIG FILTER #
    outliers_fraction = 0.05
    rng = np.random.RandomState(42)
    filter = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.11)

    filter.fit(training_features_i)
    pred_filter = filter.predict(test_features)

    # == Random Filter ==#
    rand_filter = []
    for index in range(len(pred_filter)):
        rand_filter.append(randint(0, 1))


    new_test_features = []
    new_test_labels = []
    for index in range(len(pred_filter)):
        if pred_filter[index] == 1:
            new_test_features.append(test_features[index])
            new_test_labels.append(test_labels[index])
    new_pred = extra_trees.predict(new_test_features)
    print("Accuracy After Filter - Extra Trees: {}".format(accuracy_score(new_test_labels, new_pred)))

    new_test_features = []
    new_test_labels = []
    for index in range(len(rand_filter)):
        if rand_filter[index] == 1:
            new_test_features.append(test_features[index])
            new_test_labels.append(test_labels[index])
    new_pred = extra_trees.predict(new_test_features)
    print("Accuracy After Random Filter - Extra Trees: {}".format(accuracy_score(new_test_labels, new_pred)))

    '''print("-----------------------------------------------------")
    print(pred_filter)
    print("-----------------------------------------------------")
    print(test_labels)
    print(len(test_labels))
    print("-----------------------------------------------------")
    print(new_test_labels)
    print(len(new_test_labels))'''






#create_umafall_db()
classification()
#get_confusion_matrix()
#verify_accuracy_with_noise()