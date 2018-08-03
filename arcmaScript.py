import os
from classes.converters.arcma import ARCMAConverter
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.metrics import accuracy_score
from classes.commons.utils import verify_confusion_matrix_simple
import numpy as np
from classes.commons.classification_functions import load_training_data_with_window, calculating_features
debug = True

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
def create_file_db():
    arcma.load_data()
    arcma.slice_to_training_test(0.8, 100)
    arcma.save_to_sql("arcma.db", "arcma")

def classification():
    #loading training activitie list by window
    training, test = load_training_data_with_window(arcma, "arcma.db", "arcma", "x, y, z, activity", "activity", window_len=100)

    #Calculate Features
    training_features, training_labels = calculating_features(training)
    test_features, test_labels = calculating_features(test)
    #Fit kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    extra_trees = ExtraTreesClassifier(max_depth=100, random_state=0)
    random_forest = RandomForestClassifier(max_depth=20, random_state=1)
    decision_tree = tree.DecisionTreeClassifier()

    knn.fit(training_features, training_labels)
    extra_trees.fit(training_features, training_labels)
    random_forest.fit(training_features, training_labels)
    decision_tree.fit(training_features, training_labels)

    print("Accuracy kNN: {}".format(accuracy_score(test_labels, knn.predict(test_features))))
    print("Accuracy Extra Trees: {}".format(accuracy_score(test_labels, extra_trees.predict(test_features))))
    print("Accuracy Random Forest: {}".format(accuracy_score(test_labels, random_forest.predict(test_features))))
    print("Accuracy Decision Tree: {}".format(accuracy_score(test_labels, decision_tree.predict(test_features))))


def get_confusion_matrix():

    training, test = load_training_data_with_window(arcma, "arcma.db", "arcma", "x, y, z, activity", "activity", window_len=100)
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
#classification()
get_confusion_matrix()



