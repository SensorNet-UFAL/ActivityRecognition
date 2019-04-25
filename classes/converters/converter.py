# -*- coding: utf-8 -*-
from os import listdir
import csv
import os.path
from os.path import isfile, join
import random
import sqlite3
from classes.commons.utils import print_debug, get_data_sql_query
import sys


class Converter(object):
    def load_csv_files(self, start_line, sep):
        mypath = os.path.dirname(os.path.abspath(__file__))
        datasetPath = os.path.abspath(os.path.join(mypath, self.path))
        filepaths = [join(datasetPath, f) for f in listdir(datasetPath) if isfile(join(datasetPath, f)) if
                     f.endswith(".csv")]
        file_names = [f for f in listdir(datasetPath) if isfile(join(datasetPath, f)) if f.endswith(".csv")]
        for index_path, f in enumerate(filepaths):
            # if index_path > 50:
            #	break
            print("Load csv files: " + str(round(float(index_path + 1) / float(len(filepaths)), 3) * 100) + "%")
            tempFile = open(f, "rt")
            readCSV = csv.reader(tempFile, delimiter=sep)
            csv_file = []
            #print(f)
            for index, row in enumerate(readCSV):
                if index >= start_line:
                    if index == start_line:
                        row.append("file")
                    else:
                        row.append(file_names[index_path])
                    csv_file.append(row)
            tempFile.close()
            self.csv_files.append(csv_file)


    def __init__(self, path_arg="."):
        self.csv_files = []
        self.data = {}
        self.path = path_arg
        self.readings = []
        self.traning_list = []
        self.test_list = []
        self.data_frame = {}

    #list  = list of objects to slice
    def slice_to_training_test(self, training_proportion=0.8, seed=1):

        print_debug("Calculating the size of lists...")
        list_len = len(self.readings)
        training_len = int((list_len*training_proportion))
        test_len = list_len - training_len

        #Initialzing the list of test and training

        print_debug("Initializing the sets of training and test...")
        training_list = self.readings.copy()
        test_list = []

        #List with test indexes to put in test list
        print_debug("Calculating the random indexes...")
        random.seed(seed)
        test_index = random.sample(range(training_len), test_len)

        print_debug("Loop for create the sets of training e test...")
        for index in test_index:
            test_list.append(self.readings[index])
            training_list.pop(index)

        print_debug("List len: {}".format(len(self.readings)))
        print_debug("Training len: {}".format(len(training_list)))
        print_debug("Test len: {}".format(len(test_list)))
        print_debug("Training + test len: {}".format(len(training_list)+len(test_list)))

        self.traning_list = training_list
        self.test_list = test_list

    def get_readings_by_activity(self, filename,tablename, activity, features, activity_column_name="activity"):
        dataset = sqlite3.connect(filename)
        query = "select {} from {} where {} = {} order by time".format(features, tablename, activity_column_name, activity)
        print(query)
        return get_data_sql_query(query, dataset)

    #Return all readings
    def get_all_readings(self, filename, tablename, features):
        dataset = sqlite3.connect(filename)
        return get_data_sql_query("select {} from {}".format(features, tablename), dataset)

    def get_readings_by_sql(self, filename, sql):
        dataset = sqlite3.connect(filename)
        return get_data_sql_query(sql, dataset)

    def get_all_readings_from_person(self, filename, tablename, features, person_tag, person_column, additional_where = ""):
        dataset = sqlite3.connect(filename)
        if len(additional_where) > 0:
            to_return = get_data_sql_query("select {} from {} where {} = {} {}".format(features, tablename, person_column, person_tag, additional_where), dataset)
        else:
            to_return = get_data_sql_query("select {} from {} where {} = {}".format(features, tablename, person_column, person_tag), dataset)
        return to_return

    #Do: function that return one list with activities
    def load_list_of_activities(self, filename, tablename, features, activities_indexes, separated=True, activity_column_name = "activity"):
        if separated:
            #activities_indexes =[1,2,3,4,5,6,7]
            activities_list = []
            for i in activities_indexes:
                activities_list.append(self.get_readings_by_activity(filename, tablename, i, features, activity_column_name))
            return activities_list
        else:
            return self.get_all_readings(filename, tablename, features)
