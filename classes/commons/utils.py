from operator import itemgetter
import sys

sys.path.insert(0, '../')

#--IMPORTAÇÕES DE BIBLIOTECAS AUXILIARES--#
import random
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import pylab
import time
from sklearn.metrics import accuracy_score
import itertools
import math
from scipy.stats import kurtosis, skew
from classes.commons.struct import Struct
import plotly.plotly as py
import plotly.graph_objs as go

#--IMPORTAÇÃO DE BIBLIOTECAS DE MACHINE LEARNING--#
from sklearn.model_selection import cross_val_score #Para realizar o Cross Validation
from sklearn import neighbors # KNN
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn import svm #SVM
from sklearn import tree # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees
from sklearn.neural_network import MLPClassifier # Multi-Layer Perceptron
from sklearn import linear_model # Linear Regression
#--IMPORTAÇÃO DE BIBLIOTECAS DE VISUALIZAÇÃO--#
from sklearn.metrics import confusion_matrix
#-------------------------------------------------#

# CONSTANTS
MEAN = "Mean"
SD = "SD"
SKEW = "Skew"
KURTOSIS = "Kurtosis"
IQR = "IQR"
ADM = "ADM"
ADSD = "ADSD"

LOG_FILE = "output.txt"

DEBUG = True


# ------------------------

def format_as_table(data,
                    keys,
                    header=None,
                    sort_by_key=None,
                    sort_order_reverse=False):
    if sort_by_key:
        data = sorted(data,
                      key=itemgetter(sort_by_key),
                      reverse=sort_order_reverse)

    # If header is not empty, add header to data
    if header:
        # Get the length of each header and create a divider based
        # on that length
        header_divider = []
        for name in header:
            header_divider.append('-' * len(name))

        # Create a list of dictionary from the keys and the header and
        # insert it at the beginning of the list. Do the same for the
        # divider and insert below the header.
        header_divider = dict(zip(keys, header_divider))
        data.insert(0, header_divider)
        header = dict(zip(keys, header))
        data.insert(0, header)

    column_widths = []
    for key in keys:
        column_widths.append(max(len(str(column[key])) for column in data))

    # Create a tuple pair of key and the associated column width for it
    key_width_pair = zip(keys, column_widths)

    format = ('%-*s ' * len(keys)).strip() + '\n'
    formatted_data = ''
    for element in data:
        data_to_format = []
        # Create a tuple that will be used for the formatting in
        # width, value format
        for pair in key_width_pair:
            data_to_format.append(pair[1])
            data_to_format.append(element[pair[0]])
        formatted_data += format % tuple(data_to_format)
    return formatted_data


def preprocess(dataset, features_keys, label_key, frac_train, seed):

    # create train and test
    train = dataset.sample(frac=frac_train, random_state=seed)
    test = dataset.drop(train.index)

    features_train = train[features_keys]
    features_test = test[features_keys]
    labels_train = train[label_key]
    labels_test = test[label_key]
    return features_train, features_test, labels_train, labels_test

def split_features_labels(dataset, features_keys, label_key):
    features = dataset[features_keys]
    labels_train = dataset[label_key]
    return features, labels_train

def cross_validation(classifier, dataset, features_keys, label_key, n_fold):
    data_aux = {}
    for f in features_keys:  # Init dict
        data_aux[f] = []
    for d in dataset:
        for f in features_keys:
            data_aux[f].append(d[f])
    # init dataframe
    data_fit = pd.DataFrame()
    for f in features_keys:
        data_fit[f] = data_aux[f]

    # Make the N Folds
    frac = round(float(1.0 / n_fold), 2)
    random.shuffle(data_fit)  # shuffle the dataset
    data_fit_fold = []
    current_fold = 0
    len_fold = int(len(data_fit) / n_fold)
    for n in range(n_fold):

        aux_list = []

        if current_fold < (n_fold - 1):
            for f in range(current_fold * len_fold, (current_fold + 1) * len_fold):
                aux_list.append(data_fit[f])
        else:
            for f in range(current_fold * len_fold, len(data_fit)):
                aux_list.append(data_fit[f])

        data_fit_fold.append(aux_list)
        current_fold = current_fold + 1

    # Make the Predict
    data_fit_fold_aux = data_fit_fold
    accuracy_list = []
    for index, dff in enumerate(data_fit_fold):
        test = data_fit_fold_aux.pop(index)
        train = union_list_of_list(data_fit_fold_aux)
        features_train = train.drop(label_key, 1)
        features_test = test.drop(label_key, 1)
        labels_train = train[label_key]
        labels_test = test[label_key]

        classifier.fit(features_train, labels_train)
        pred = classifier.predict(features_test)
        accuracy_list.append(accuracy_score(labels_test, pred))  # insert the accuracy in list of accuracies

        # Restore original list
        data_fit_fold_aux.insert(index, test)

    return accuracy_list


def preprocess_for_pca(dataset, features_keys, label_key, frac_train):
    return ""


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def slice_features(features, length):
    cont_aux = 0
    list_aux = []
    list_out = []
    for f in features:
        list_aux.append(f)
        cont_aux += 1
        if cont_aux == (length):
            list_out.append(list_aux)
            list_aux = []
            cont_aux = 0
    return list_out


def get_measures(features_list, features_keys, label_key, measure):
    m_list = []
    for fl in features_list:
        aux_list = {}
        for fk in features_keys:
            aux = []
            for f in fl:
                if fk is not label_key:
                    aux.append(float(f[fk]))
            if fk is not label_key:
                if measure == MEAN:
                    aux_list[MEAN + "_" + fk] = np.mean(aux)
                elif measure == SD:
                    aux_list[SD + "_" + fk] = np.std(aux)
                elif measure == SKEW:
                    aux_list[SKEW + "_" + fk] = skew(aux)
                elif measure == KURTOSIS:
                    aux_list[KURTOSIS + "_" + fk] = kurtosis(aux)
                elif measure == IQR:
                    aux_list[IQR + "_" + fk] = np.subtract(*np.percentile(aux, [75, 25]))
                elif measure == ADM:
                    aux_list[ADM + "_" + fk] = absolute_difference_list(aux, MEAN)
                elif measure == ADSD:
                    aux_list[ADSD + "_" + fk] = absolute_difference_list(aux, SD)
                else:
                    print("None measure!")
                    aux_list[fk] = np.mean(aux)
            else:
                aux_list[fk] = f[fk]
        m_list.append(aux_list)

    return m_list


def absolute_difference_list(list_in, method):
    list_out = []
    for index, l in enumerate(list_in):
        if index > 0:
            list_out.append(math.fabs(l - list_in[index - 1]))
    if method == MEAN:
        return np.mean(list_out)
    if method == SD:
        return np.std(list_out)
    return list_out


def union_objects_in_list(list_object_1, list_object_2):
    for index, l in enumerate(list_object_1):
        l.update(list_object_2[index])
    return list_object_1


def dataset_to_iris_format(dataset, features_keys, label_key):
    data_list = []
    target_list = []
    for d in dataset:
        aux = []
        for f in features_keys:
            aux.append(d[f])
        target_list.append(d[label_key])
        data_list.append(aux)
    return Struct({"data": data_list, "target": target_list})


def union_list_of_list(list_in):
    list_out = []
    for l in list_in:
        list_out = list_out + l
    return list_out

def plot_table_dataframe(df):
    trace = go.Table(
        header=dict(values=df.columns,
                    fill=dict(color='#C2D4FF'),
                    align=['left'] * 5),
        cells=dict(values=df.columns,
                   fill=dict(color='#F5F8FF'),
                   align=['left'] * 5))
    data = [trace]
    py.iplot(data, filename='pandas_table')

def split_dataframe_by_timestamp(data, timestamp_label, timestamp_interval):
    dataframe_final = pd.DataFrame(columns=["Mean_X","Mean_Y","Mean_Z"])
    dataframe_aux = pd.DataFrame(columns=data.columns)

    last_timestamp = None
    for index, row in data.iterrows():
        # Iniciando a verificação do TimeStamp
        if last_timestamp is None:
            last_timestamp = row[timestamp_label]
        if(row[timestamp_label] - last_timestamp > timestamp_interval):
            print("Len: {}".format(len(dataframe_aux)))
            last_timestamp = row[timestamp_label]
            dataframe_aux = pd.DataFrame(columns=data.columns)

        else:
            dataframe_aux.loc[len(dataframe_aux)] = row


    return ""

#Test specific algorithm with cross validation
def verify_accuracy_cross_validation(data, classifier, features_keys, label_key,n_fold):
    features, labels = split_features_labels(data, features_keys, label_key)
    t = time.time()
    scores = cross_val_score(classifier, features, labels, cv=n_fold)
    score = np.mean(scores)
    timer = round(float(time.time() - t), 3)
    print("Score {} = {}, Tempo = {}".format(classifier.__class__.__name__, score, timer))

#Test machinelearning algorithms with all dataset
def verify_accuracy_cross_validation_all_classify(con_sql, sqlite_dataset_name, selected_columns, person_len, features_keys, label_key,n_fold):
    results = []
    for person in range(1,person_len+1):
        file_print("==========================================PERSON {}======================================".format(person), LOG_FILE, True)
        for position, position_value in UmaAdlConverter.SENSOR.__dict__.items(): # Para cada dispositivo
            for sensor, sensor_value in UmaAdlConverter.SENSORTYPE.__dict__.items(): # Para cada sensor (giroscópio, acelerômetro, magnetômetro)
                consulta = "Select {} from {} where person = {} and SensorType = {} and SensorID = {} order by TimeStamp".format(
                        ", ".join(selected_columns), sqlite_dataset_name, person, sensor_value, position_value)
                file_print("+ "+consulta, LOG_FILE, True)
                all_data = get_data_sql_query(consulta, con_sql)
                if len(all_data) < 1:
                    file_print("* ATENÇÃO NENHUMA LINHA PARA: "+consulta, LOG_FILE, True)
                    continue
                features, labels = split_features_labels(all_data, features_keys, label_key)
                classifiers = []
                classifiers.append(neighbors.KNeighborsClassifier(n_neighbors=5))
                classifiers.append(GaussianNB())
                classifiers.append(tree.DecisionTreeClassifier())
                classifiers.append(RandomForestClassifier(max_depth=20, random_state=1))
                classifiers.append(ExtraTreesClassifier(max_depth=100, random_state=0))
                #classifiers.append(svm.SVC())
                for clf in classifiers:
                    t = time.time()
                    scores = cross_val_score(clf, features, labels, cv=n_fold)
                    score = np.mean(scores)
                    sd = np.std(scores)
                    timer = round(float(time.time()-t), 3)
                    file_print("Pessoa {} - Position: {} - Sensor: {} - Metodo: {} - Score: {} - SD: {} - Time: {}s".format(person, position, sensor, clf.__class__.__name__, score, sd, timer), LOG_FILE, True)
                    results.append({"classifier":clf.__class__.__name__, "person": person, "sensorName": sensor, "position": position, "dataset_len": len(all_data),"activity_len": len(all_data.activity.unique()), "sql": consulta, "score": score, "sd": sd, "time": timer})

    #Salvando a saída no banco de dados.
    dataset = sqlite3.connect("output.db")
    df_results = pd.DataFrame(results)
    df_results.to_sql("output", dataset,if_exists='replace', index=False)
    file_print("================Finish verify accuracy=====================", LOG_FILE, True)

def plot_accuracy_by_position(con_sql, classifier, sqlite_dataset_name, selected_columns, person):
    colors = {"ACCELEROMETER":"b", "GYROSCOPE":"g", "MAGNETOMETER":"r"}
    #translate_sensor = {"ACCELEROMETER": "Acelerômetro", "GYROSCOPE": "Giroscópio", "MAGNETOMETER": "Magnetômetro"}
    #translate_position = {"CHEST":"Torax", "WRIST":"Pulso", "ANKLE":"Tornozelo", "WAIST":"Cintura", "RIGHTPOCKET":"Bolso Direito"}
    translate_sensor = {"ACCELEROMETER": "Accelerometer", "GYROSCOPE": "Gyroscope", "MAGNETOMETER": "Magnetometer"}
    translate_position = {"CHEST":"Chest", "WRIST":"Wrist", "ANKLE":"Ankle", "WAIST":"Waist", "RIGHTPOCKET":"Right Pocket"}

    for sensor, sensor_value in UmaAdlConverter.SENSORTYPE.__dict__.items():  # Para cada sensor (giroscópio, acelerômetro, magnetômetro)
        x = []
        y = []
        for position, position_value in UmaAdlConverter.SENSOR.__dict__.items():  # Para cada dispositivo
            consulta = "Select {} from {} where classifier = '{}' and person = '{}' and sensorName = '{}' and position = '{}'".format(
                ", ".join(selected_columns), sqlite_dataset_name, classifier, person, sensor, position)
            print(consulta)
            all_data = get_data_sql_query(consulta, con_sql)
            if(len(all_data)>0):
                x.append(translate_position[all_data.position[0]])
                y.append(int(1000*round(all_data.score[0], 3))/10.0)
                #plt.bar(all_data.position[0], all_data.score[0], width=0.5, align='center', color=colors[sensor], label= translate[sensor])
        plt.bar(x, y, width=0.5, align='center', color=colors[sensor],
                label=translate_sensor[sensor])
        for i in range(len(x)):
            plt.text(x[i], y[i] + 0.2, s="{}%".format(y[i]), size=10)

    plt.legend(loc=3)
    plt.subplots_adjust(bottom=0.2, top=0.98)
    plt.savefig('accuracy_by_position.png')

def verify_confusion_matrix(classifier, data, features_keys, label_key, title):
    features_train, features_test, labels_train, labels_test = preprocess(data, features_keys, label_key, 0.9, 50)
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred, labels=np.unique(labels_test))
    np.set_printoptions(precision=2)
    class_names = np.unique(labels_test)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, class_names, True, title=title)
    plt.savefig("confusion_matrix_extra_trees.png")

def connect_sql(filename):
    sqlite3.connect(filename)

def get_data_sql_query(query,dataset):
    return pd.read_sql_query(query, dataset)

def file_print(text, output, p=False):
    if p:
        print(text)
    file = open(output, 'a')
    file.write(text+"\n")
    file.close()

def translate_activities(data, activity_label):
    data[activity_label] = data[activity_label].replace("Bending", "Inclinar-se")
    data[activity_label] = data[activity_label].replace("Hopping", "Saltar")
    data[activity_label] = data[activity_label].replace("Jogging", "Correr")
    data[activity_label] = data[activity_label].replace("LyingDown", "Deitar")
    data[activity_label] = data[activity_label].replace("Sitting", "Sentar")
    data[activity_label] = data[activity_label].replace("Walking", "Caminhar")
    data[activity_label] = data[activity_label].replace("backwardFall", "Queda trás")
    data[activity_label] = data[activity_label].replace("forwardFall", "Queda frontal")
    data[activity_label] = data[activity_label].replace("lateralFall", "Queda lateral")

def plot_accuracy_by_algorithm(x,y, x_label, y_label):
    plt.bar(x, y, width=0.50, align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(len(x)):
        plt.text(x[i], y[i]+0.2, s="{}%".format(y[i]), size=10)
    plt.subplots_adjust(bottom=0.2, top=0.98)
    plt.savefig('accuracy_by_algorithm.png')

def plot_time_by_algorithm(x,y, x_label, y_label):
    plt.bar(x, y, width=0.50, align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(len(x)):
        plt.text(x[i], y[i]+0.2, s="{}s".format(y[i]), size=10)
    plt.subplots_adjust(bottom=0.2, top=0.98)
    plt.savefig('time_by_algorithm.png')


def knn_verify_accuracy_cross_validation(data, features_keys, label_key,n_fold, test_k,show_matrix):
    features_all, labels_all = split_features_labels(data, features_keys, label_key)
    features_train, features_test, labels_train, labels_test = preprocess(data, features_keys, label_key, 0.8, 50)

    if test_k:
        a = []
        r = range(1, 201)
        for i in r:
            clf = neighbors.KNeighborsClassifier(n_neighbors=i)
            # --CROSS VALIDATION--#
            scores = cross_val_score(clf, features_all, labels_all, cv=n_fold)
            a.append(np.mean(scores))
        plt.plot(r, a)
        plt.show()
    else:
        clf = neighbors.KNeighborsClassifier(n_neighbors=5)

        #--CROSS VALIDATION--#
        scores = cross_val_score(clf, features_all, labels_all, cv=n_fold)
        print("- Knn: {}".format(np.mean(scores)))

    #--CONFUSION MATRIX--#
    if show_matrix:
        clf.fit(features_train,labels_train)
        pred = clf.predict(features_test)
        cnf_matrix = confusion_matrix(labels_test, pred, labels=np.unique(labels_test))
        np.set_printoptions(precision=2)
        class_names = np.unique(labels_test)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, class_names, True, title='Confusion matrix, for Knn')
        plt.show()

def print_debug(msg):
    if DEBUG:
        print(msg)