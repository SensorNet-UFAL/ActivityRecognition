###### IMPORTS ######
from sklearn import svm
import numpy as np
from classes.converters.umaAdl import UmaAdlConverter
from classes.commons.classification_functions import calculating_features, load_training_data_with_window_from_sql
from sklearn.cluster import KMeans

#LOADING ALL DATA AND CREATING ONE SET TO EACH PERSON
def load_train_test_outlier_for_each_person():
    umafall = UmaAdlConverter('')
    list_train_features = []
    list_train_labels = []
    list_test_features = []
    list_test_labels = []
    for p in range(1, 3): # loop total = (1,18)
        training, test = load_training_data_with_window_from_sql(umafall, "..\\..\\umafall.db", "Select XAxis, YAxis, ZAxis, activity, person from umafall where SensorType=0 and SensorID=0 and person={}".format(p), "activity", 50)
        training_features, training_labels = calculating_features(training, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        test_features, test_labels = calculating_features(test, x_label="XAxis", y_label="YAxis", z_label="ZAxis")
        list_train_features.append(training_features)
        list_train_labels.append(training_labels)
        list_test_features.append(test_features)
        list_test_labels.append(test_labels)
    return list_train_features, list_train_labels, list_test_features, list_test_labels

def clusteringWithKMeans():
    list_train_features, list_train_labels, list_test_features, list_test_labels = load_train_test_outlier_for_each_person()
    kmeans = KMeans(n_clusters=9, random_state=0).fit(list_train_features[0])
    #Print number of activities
    print("Number of activities: {}".format(len(np.unique(list_train_labels[0]))))
    #Print number of clusters
    labels = kmeans.labels_
    train_labels = list_train_labels[0]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for l in range(1,n_clusters+1):
        indexes = np.where(labels == l)
        print("========= Cluster {} ========".format(l))
        cluster_aux = labels[indexes]
        train_labels_aux = train_labels[indexes]
        unique_activities = np.unique(train_labels_aux)
        activities_proportions = []
        for ua in unique_activities:
            ap = np.where(train_labels_aux == ua)
            p = (len(ap[0])/len(train_labels_aux))*100
            print("Activities found: {} - {}%".format(ua, p))
        print("---------------------------------")


clusteringWithKMeans()