#===== Classification Tools =====#
from classes.commons.classification_functions import load_train_test_outlier_for_each_person
from classes.converters.arcma import ARCMAConverter
from classes.converters.umaAdl import UmaAdlConverter
from classes.converters.hmp import HMPConverter
#===============================================================#

#===== Machine Learn =====#
from sklearn import svm
#===============================================================#

#===== Utils =====#
import numpy as np
#===============================================================#

#==== FUNCTIONS ====#

def get_sets_to_outlier_test(list_train_labels, list_test_labels, activity):

    train_indexes = np.where(list_train_labels != activity)
    test_indexes = np.where(list_test_labels != activity)
    outliers_indexes = np.where(list_train_labels == activity)

    return train_indexes, test_indexes, outliers_indexes

#Getting total accuracy for a features set.
def total_accuracy_for_set(features, list_train_features, list_train_labels, list_test_features, list_test_labels, activity_list):

    train_accuracy = 0
    test_accuracy = 0
    outliers_accuracy = 0

    count_loop = 0

    for i in activity_list:
        for p in range(0, len(list_train_features)):

            train_indexes, test_indexes, outliers_indexes = get_sets_to_outlier_test(list_train_labels[p], list_test_labels[p], i)
            if len(train_indexes) > 0:
                #print("%%% ENTROU %%%")
                train = list_train_features[p][train_indexes][:, features]
                test = list_test_features[p][test_indexes][:, features]
                if len(train) > 0:
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
                else:
                    print("Train set with 0 dim: {}".format(train.shape))
    '''print("========= ARCMA ===========")
    print("Train Accuracy Mean = {}%".format(train_accuracy/ count_loop))
    print("Test Accuracy Mean = {}%".format(test_accuracy / count_loop))
    print("Outliers Accuracy Mean = {}%".format(outliers_accuracy / count_loop))'''
    if count_loop > 0:
        return train_accuracy/count_loop, test_accuracy/count_loop, outliers_accuracy/count_loop
    else:
        return 0,0,0


#Finding the best combination for 3 features
def find_the_best_set_depth_3_activity_loop(activity_list, dataset, filename, tablename, features, label_axis, label_column, window_len, person_indexes, person_column, additional_where = ""):

    list_train_features, list_train_labels, list_test_features, list_test_labels = load_train_test_outlier_for_each_person(dataset=dataset, filename=filename, tablename=tablename, features=features, label_axis=label_axis, label_column=label_column, window_len=window_len, person_indexes=person_indexes, person_column=person_column, additional_where=additional_where)
    print("##########################")
    print("###### TOTAL OF PERSONS: {} ######".format(len(list_train_features)))
    print("##########################")
    #Verificar resultado do outlier para cada trio possível de feature
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
                if(train_accuracy > 60 and test_accuracy > 60 and outliers_accuracy > 60):
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
    if len(features_flag)>0:
        print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], train_accuracy_flag))
        print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], test_accuracy_flag))
        print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], outliers_accuracy_flag))
    else:
        print("Sem resultados relavantes.")

##= ARCMA =##
#arcma = ARCMAConverter("..\\..\\\\databases\\arcma")
#find_the_best_set_depth_3_activity_loop(activity_list=[1, 4, 7], dataset=arcma, filename="..\\arcma.db", tablename="arcma", features="x, y, z, activity", label_axis=["x","y","z"], label_column="activity", window_len=100, person_indexes=list(range(1,16)), person_column="person")

##= UMAFALL =##
umafall = UmaAdlConverter('')
activity_list = ["Bending","forwardFall","Hopping", "Jogging", "LyingDown", "Sitting", "Walking", "backwardFall", "lateralFall", "GoDownstairs", "GoUpstairs"]
find_the_best_set_depth_3_activity_loop(activity_list=activity_list, dataset=umafall, filename="..\\umafall.db", tablename="umafall", features="XAxis, YAxis, ZAxis, activity", label_axis=["XAxis","YAxis","ZAxis"], label_column="activity", window_len=100, person_indexes=list(range(1,18)), person_column="person", additional_where = "and SensorType=0 and SensorID=1")

##= HMP =##
hmp = HMPConverter("")
#activity_list = ["brush_teeth", "climb_stairs", "descend_stairs", "eat_meat", "eat_soup", "getup_bed", "liedown_bed", "pour_water", "standup_chair", "use_telephone", "walk"]
#person_indexes = ["'f1'", "'m1'", "'m2'", "'f2'", "'m3'", "'f3'", "'m4'", "'m5'", "'m6'", "'m7'", "'f4'", "'m8'", "'m9'", "'f5'", "'m10'", "'m11'", "'f1_1'", "'f1_2'", "'f1_3'", "'f1_4'", "'f1_5'", "'m1_1'", "'m1_2'", "'m2_1'", "'m2_2'", "'f3_1'", "'f3_2'"]
#find_the_best_set_depth_3_activity_loop(activity_list=activity_list, dataset=hmp, filename="..\\hmp.db", tablename="hmp", features="x, y, z, activity", label_axis=["x","y","z"], label_column="activity", window_len=100, person_indexes=person_indexes, person_column="person", additional_where = "")
#clusteringWithoutSimilarActivities(hmp, 14,"..\\hmp.db", "hmp", "x, y, z, activity", ["x","y","z"], "activity", 100, person_indexes, "person", verbose=True)

