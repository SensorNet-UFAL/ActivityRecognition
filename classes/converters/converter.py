# -*- coding: utf-8 -*-
from os import listdir
import csv
import os.path
from os.path import isfile, join
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


    def __init__(self, path_arg):
        self.csv_files = []
        self.data = {}
        self.path = path_arg
        self.readings = []
