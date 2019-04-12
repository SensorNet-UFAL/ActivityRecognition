#===== Classification Tools =====#
from classes.commons.classification_functions import load_training_data_with_window_from_person, calculating_features
from classes.converters.hmp import HMPConverter
#===== Machine Learn =====#
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn import svm #SVM
#===== Utils =====#
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
#===============================================================#

#==== Global Variables ====#
most_accurate_person = {"clf": "", "id": 0, "accuracy": 0}
#==============================================================#

#==== FUNCTIONS ====#
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
    if accuracy > most_accurate_person["accuracy"]:
        most_accurate_person['accuracy'] = accuracy
        most_accurate_person['id'] = clf_store['person']
        most_accurate_person['clf'] = clf_store['name']

def print_time_to_fit(clf_store):
    print("---------------------------------------")
    print("Classifier: {}".format(clf_store['name']))
    print("Time to fit: {}s".format(clf_store['time_fit'] / clf_store['len']))

def print_accuracy(clf_store):
    print("Time to predict: {}s".format(clf_store['time_classify']/clf_store['len']))
    print("Accuracy: {}".format(clf_store['accuracy'] / clf_store['len']))
    print("F-score: {}".format(clf_store['f1'] / clf_store['len']))
    print("Most Accuracy: CLF: {}, Person: {}, Accuracy: {}".format(most_accurate_person['clf'], most_accurate_person['id'], most_accurate_person['accuracy']))


def classification(dataset, filename, tablename, features, label, window_len, person_indexes, person_column):
    # --init classifier temp--#
    knn_store = {"name": "KNN", "person":"","clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0, "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    nb_store = {"name": "Naive Bayes", "person":"", "clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0, "time_classify": 0,
                "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    extra_trees_store = {"name": "Extra-Trees", "person":"", "clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0,
                         "time_classify": 0,
                         "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    random_forest_store = {"name": "Random Forest", "person":"", "clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0,
                           "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None,
                           "test_labels": None}
    decision_tree_store = {"name": "Decision Tree", "person":"", "clf": None, "len": 0, "accuracy": 0, "f1": 0, "time_fit": 0,
                           "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None,
                           "test_labels": None}
    svm_store = {"name": "SVM", "person":"", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "f1": 0, "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}

    for person_index in person_indexes:
        training, test = load_training_data_with_window_from_person(dataset, filename, tablename, features,
                                                                    label, window_len, person_index, person_column)
        training_features, training_labels = calculating_features(training)
        test_features, test_labels = calculating_features(test)

        print("##PERSON ACTIVITY = {}".format(person_index))

        if len(training_labels) > 10 and len(np.unique(training_labels)) > 1:
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

#==== CALL FUNCTIONS ====#

#==== HMP Dataset ====#
hmp = HMPConverter("")
person_indexes = ["'f1'", "'m1'", "'m2'", "'f2'", "'m3'", "'f3'", "'m4'", "'m5'", "'m6'", "'m7'", "'f4'", "'m8'", "'m9'", "'f5'", "'m10'", "'m11'", "'f1_1'", "'f1_2'", "'f1_3'", "'f1_4'", "'f1_5'", "'m1_1'", "'m1_2'", "'m2_1'", "'m2_2'", "'f3_1'", "'f3_2'"]
classification(hmp, "..\\hmp.db", "hmp", "x, y, z, activity", "activity", 100,person_indexes, "person")
