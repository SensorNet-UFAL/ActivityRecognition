#Dependences
#pip install tabulate
#pip install pandas
#pip install sklearn
#pip install scipy
#pip install matplotlib

# -*- coding: utf-8 -*-

#--IMPORTAÇÕES DE BIBLIOTECAS AUXILIARES--#
from classes.converters.umaAdl import UmaAdlConverter
from classes.commons.utils import *
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import pylab
import time
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

##===== CONSTANTES =====##
SQLITE_FILENAME = "umafall_dataset.db"
SQLITE_DATASET_NAME = "umafall"
LOG_FILE = "output.txt"
##======================##


###### === SCRIPT === ######

#--PARA CRIAR O ARQUIVO SQLITE EXECUTAR ESSE COMANDO--#
#convert_umafall_from_csv_to_sqlite(SQLITE_FILENAME, SQLITE_DATASET_NAME)

#--CRIANDO A CONEXÃO COM O BANCO DE DADOS E CONFIGURAÇÃO INICIAL--#
con_dataset = sqlite3.connect(SQLITE_FILENAME)
selected_columns = ["TimeStamp","XAxis", "YAxis", "ZAxis", "activity"]
features = ["XAxis", "YAxis", "ZAxis"]
label = selected_columns[-1]
#-----------------------------------------------------------------#


#--TESTANDO KNN--#
#all_data = get_data_sql_query("select {} from {} where person = 1 and SensorType = 2 order by TimeStamp".format(", ".join(selected_columns), SQLITE_DATASET_NAME), con_dataset)
#knn_verify_accuracy_cross_validation(all_data,features, label, 10, False, True)
#plt.plot(all_data["TimeStamp"], all_data["XAxis"],c=all_data["activity"])
#plt.show()

#--SCORES PARA TODOS OS METODOS--##
#verify_accuracy_cross_validation_all_classify(con_dataset, SQLITE_DATASET_NAME, selected_columns, 18, features, label, 10)

#--SCORE PARA O SVM--##
#data = get_data_sql_query("Select TimeStamp, XAxis, YAxis, ZAxis, activity from umafall where person = 15 and SensorType = 2 and SensorID = 2 order by TimeStamp", con_dataset)
#clf = svm.SVC()
#verify_accuracy_cross_validation(data, clf, features, label, 10)

#--VERIFICANDO RESULTADO COM MATRIZ DE CONFUSÃO--#
#data = get_data_sql_query("Select TimeStamp, XAxis, YAxis, ZAxis, activity from umafall where person = 15 and SensorType = 2 and SensorID = 2 order by TimeStamp", con_dataset)
#classifier = ExtraTreesClassifier(max_depth=100, random_state=0)
#translate_activities(data, "activity")
#verify_confusion_matrix(classifier, data, features, label, "ExtraTrees - Matríz de confusão.")

#--IMPRIMINDO HISTOGRAMA DE ACURÁCIA--#
#x = ["k-NN", "NB", "DT", "RF", "ET", "SVM","RNN"]
#y = [89.7, 59.9, 88.2, 89.3, 89.9, 82.2, 70.62]
#plot_accuracy_by_algorithm(x, y)

#--IMPRIMINDO TEMPO DOS ALGORITMOS--#
#x = ["k-NN", "NB", "DT", "RF", "ET", "SVM", "RNN"]
#y = [0.216, 0.181, 0.234, 0.646, 0.544, 22.66, 1620.0]
x = ["k-NN", "NB", "DT", "RF", "ET", "SVM"]
y = [0.216, 0.181, 0.234, 0.646, 0.544, 22.66]
plot_time_by_algorithm(x, y)

#--IMPRIMINDO ACURÁCIA POR SENSOR--#
#con_output = sqlite3.connect("output.db")
#plot_accuracy_by_position(con_output, "ExtraTreesClassifier", "output", "*", 15)
#--CALCULANDO CARACTERÍSTICAS--#
#utils.split_dataframe_by_timestamp(all_data, "TimeStamp", 50)


