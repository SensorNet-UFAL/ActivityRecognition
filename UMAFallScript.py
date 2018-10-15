
from classes.converters.umaAdl import UmaAdlConverter
from classes.commons.classification_functions import load_training_data_with_window_from_sql

def create_umafall_db():
    umafall = UmaAdlConverter('')
    umafall.load_from_csv("C:\\Users\\wylken.machado.INTRA\\ownCloud\\WYLKEN\\MESTRADO\\2016\\Dissertacao\\Implementacoes\\Projeto_Artigo\\Datasets\\UMA_ADL_FALL_Dataset")
    umafall.convert_csv_to_sql("umafall.db", "umafall")



def classification():
    umafall = UmaAdlConverter('')

    # --init classifier temp--#
    knn_store = {"name": "KNN", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    extra_trees_store = {"name": "Extra-Trees", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                         "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}
    random_forest_store = {"name": "Random Forest", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0,
                           "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None,
                           "test_labels": None}
    decision_tree_store = {"name": "Decision Tree", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0,
                           "time_classify": 0,
                           "training_features": None, "training_labels": None, "test_features": None,
                           "test_labels": None}
    svm_store = {"name": "SVM", "clf": None, "len": 0, "accuracy": 0, "time_fit": 0, "time_classify": 0,
                 "training_features": None, "training_labels": None, "test_features": None, "test_labels": None}

    for person_index in range(1, 18):
        training, test = load_training_data_with_window_from_sql(umafall, "umafall.db", "Select XAxis, YAxis, ZAxis, activity from umafall where person={}".format(person_index), "activity", 100)


#create_umafall_db()
classification()