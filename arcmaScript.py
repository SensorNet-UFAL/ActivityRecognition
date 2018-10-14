import os
from classes.converters.arcma import ARCMAConverter
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn import svm #SVM
from sklearn.metrics import accuracy_score
from classes.commons.utils import verify_confusion_matrix_simple
import time
import numpy as np
from classes.commons.classification_functions import load_training_data_with_window, calculating_features, load_training_data_with_window_from_person
debug = True

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
def create_file_db():
    arcma.load_data()
    #arcma.slice_to_training_test(0.8, 100)
    arcma.save_to_sql("arcma.db", "arcma")


def put_clf_init_values(clf_store, clf, training_features, training_labels, test_features, test_labels):
    clf_store['clf'] = clf
    clf_store['training_features'] = training_features
    clf_store['training_labels'] = training_labels
    clf_store['test_features'] = test_features
    clf_store['test_labels'] = test_labels

def verify_fit(clf_store):
    start = time.time()
    clf = clf_store['clf']
    clf.fit(clf_store['training_features'], clf_store['training_labels'])
    elapsed = time.time() - start
    clf_store['time_fit'] += elapsed

def verify_accuracy(clf_store):
    start = time.time()
    clf = clf_store['clf']
    clf_store['accuracy'] += accuracy_score(clf_store['test_labels'], clf.predict(clf_store['test_features']))
    elapsed = time.time() - start
    clf_store['time_classify'] += elapsed
    clf_store['len'] += 1

def print_accuracy(clf_store):
    print("Time to predict: {}s".format(clf_store['time_classify']/clf_store['len']))
    print("Accuracy: {}s".format(clf_store['accuracy'] / clf_store['len']))

def print_time_to_fit(clf_store):
    print("---------------------------------------")
    print("Classifier: {}".format(clf_store['name']))
    print("Time to fit: {}s".format(clf_store['time_fit'] / clf_store['len']))

def classification():

    #--init classifier temp--#
    knn_store = {"name": "KNN", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    extra_trees_store = {"name": "Extra-Trees", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                         "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    random_forest_store = {"name": "Random Forest", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    decision_tree_store = {"name": "Decision Tree", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    svm_store = {"name": "SVM", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}


    for person_index in range(1,16):
        training, test = load_training_data_with_window_from_person(arcma, "arcma.db", "arcma", "x, y, z, activity",
                                                                    "activity", 100, person_index, "person")
        training_features, training_labels = calculating_features(training)
        test_features, test_labels = calculating_features(test)

        #==KNN=#
        knn = KNeighborsClassifier(n_neighbors=21)
        put_clf_init_values(knn_store, knn, training_features, training_labels, test_features, test_labels)
        verify_fit(knn_store)
        verify_accuracy(knn_store)

        #==Extra-Trees==#
        extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
        put_clf_init_values(extra_trees_store, extra_trees, training_features, training_labels, test_features, test_labels)
        verify_fit(extra_trees_store)
        verify_accuracy(extra_trees_store)

        #==Random Forest==#
        random_forest = RandomForestClassifier(max_depth=20, random_state=1)
        put_clf_init_values(random_forest_store, random_forest, training_features, training_labels, test_features, test_labels)
        verify_fit(random_forest_store)
        verify_accuracy(random_forest_store)

        #==Decision Tree==#
        decision_tree = tree.DecisionTreeClassifier()
        put_clf_init_values(decision_tree_store, decision_tree, training_features, training_labels, test_features, test_labels)
        verify_fit(decision_tree_store)
        verify_accuracy(decision_tree_store)

    print_time_to_fit(knn_store)
    print_accuracy(knn_store)

    print_time_to_fit(extra_trees_store)
    print_accuracy(extra_trees_store)

    print_time_to_fit(random_forest_store)
    print_accuracy(random_forest_store)

    print_time_to_fit(decision_tree_store)
    print_accuracy(decision_tree_store)


    '''
    #loading training activitie list by window
    #training, test = load_training_data_with_window(arcma, "arcma.db", "arcma", "x, y, z, activity", [1,2,3,4,5,6,7],"activity", window_len=100)
    training, test = load_training_data_with_window_from_person(arcma, "arcma.db", "arcma", "x, y, z, activity", "activity", 100, 2,"person")
    #Calculate Features
    training_features, training_labels = calculating_features(training)
    test_features, test_labels = calculating_features(test)
    #Start Clessifiers
    knn = KNeighborsClassifier(n_neighbors=21)
    extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
    random_forest = RandomForestClassifier(max_depth=20, random_state=1)
    decision_tree = tree.DecisionTreeClassifier()
    svm_clf = svm.SVC()

    #Fit Classifiers
    print_time_to_fit("KNN", knn, training_features, training_labels)
    print_accuracy("Accuracy kNN: {}", knn, test_labels, test_features)
    #-----------
    print_time_to_fit("Extra Trees", extra_trees, training_features, training_labels)
    print_accuracy("Accuracy Extra Trees: {}", extra_trees, test_labels, test_features)
    #-----------
    print_time_to_fit("Random Forest", random_forest, training_features, training_labels)
    print_accuracy("Accuracy Random Forest: {}", random_forest, test_labels, test_features)
    # -----------
    print_time_to_fit("Decision Tree", decision_tree, training_features, training_labels)
    print_accuracy("Accuracy Decision Tree: {}", decision_tree, test_labels, test_features)
    # -----------
    print_time_to_fit("SVM", svm_clf, training_features, training_labels)
    print_accuracy("Accuracy SVM: {}", svm_clf, test_labels, test_features)
    '''

def get_confusion_matrix():

    training, test = load_training_data_with_window(arcma, "arcma.db", "arcma", "x, y, z, activity",[1,2,3,4,5,6,7], "activity", window_len=100)
    # Calculate Features
    training_features, training_labels = calculating_features(training)
    test_features, test_labels = calculating_features(test)

    extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
    extra_trees.fit(training_features, training_labels)
    pred = extra_trees.predict(test_features)
    print("Accuracy Extra Trees: {}".format(accuracy_score(test_labels, pred)))
    verify_confusion_matrix_simple(test_labels, pred, "Matriz de Confusão - Extre Trees.")

    random_forest = RandomForestClassifier(max_depth=20, random_state=1)
    random_forest.fit(training_features, training_labels)
    pred = random_forest.predict(test_features)
    print("Accuracy Random Forest: {}".format(accuracy_score(test_labels, pred)))
    verify_confusion_matrix_simple(test_labels, pred, "Matriz de Confusão - Random Forest.")

#activity_raw = arcma.get_readings_by_activity("arcma.db", 1, "x")
#activity_windows = slice_by_window(activity_raw, 50)
#dtw_mean = get_dtw_mean(activity_windows, sample)
#print("Mean1 = {}".format(dtw_mean))

#create_file_db()
classification()
#get_confusion_matrix()



