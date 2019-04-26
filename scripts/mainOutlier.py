#===== Classification Tools =====#
#from mainClustering import load_train_test_outlier_for_each_person
#===============================================================#

#===== Machine Learn =====#

#===============================================================#

#===== Utils =====#

#===============================================================#

#==== FUNCTIONS ====#



#DEPTH 3 WITH ACTIVITY LOOP
def find_the_best_set_depth_3_activity_loop(activity_list):

    list_train_features, list_train_labels, list_test_features, list_test_labels = load_all_data_to_outlier_test_umafall()
    print("##########################")
    print("###### TOTAL OF PERSONS: {} ######".format(len(list_train_features)))
    print("##########################")
    #Verificar resultado do outlier para cada dupla possível de feature
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
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], train_accuracy_flag))
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], test_accuracy_flag))
    print("Best Train Accuracy - columns [{}], [{}], [{}]: {}%".format(features_flag[0], features_flag[1], features_flag[2], outliers_accuracy_flag))
