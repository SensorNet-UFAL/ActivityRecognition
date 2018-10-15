# -*- coding: utf-8 -*-
import sys
from classes.converters.converter import Converter
from classes.commons.person import Person
from classes.commons.struct import Struct
import pandas as pd
import sqlite3
import pickle

file_save_persons = "umaAdl.pkl"


class UmaAdlConverter(Converter):
    # Config convert
    row_filename_index = 7
    char_filename_split = '_'
    subject_name_index = 2
    subject_activity_index = 4

    # Statical Variables
    SENSOR = Struct({"RIGHTPOCKET": 0, "CHEST": 1, "WRIST": 3, "ANKLE": 4, "WAIST": 2})
    KEYS = Struct({"X": "X-Axis", "Y": "Y-Axis", "Z": "Z-Axis", "SENSORTYPE": "SensorType", "TIMESTAMP": "TimeStamp",
                   "FILE": "file", "ACTIVITY": "activity", "SENSORID": "SensorID"})
    SENSORTYPE = Struct({"MAGNETOMETER": 2, "ACCELEROMETER": 0, "GYROSCOPE": 1})

    def load_data(self):
        Converter.load_csv_files(self, 40, ";")  # load csv data in many files
        features_list = []

        last_person = None
        for index_csv, csv_file in enumerate(self.csv_files):
            for index_row, row in enumerate(csv_file):
                reading = {}
                #Resgatando o nome das colunas
                if index_row == 0 and index_csv == 0:
                    row.pop(len(row) - 2)
                    for f in row:  # start list of the features
                        f = f.replace(" ", "")
                        f = f.replace("%", "")
                        f = f.replace("-", "")
                        features_list.append(f)
                    print(features_list)
                elif index_row == 0 and index_csv > 0:
                    "Next file!"
                #Resgatando os valores
                else:
                    for index_f, f in enumerate(row):
                        try:
                            reading[features_list[index_f]] = float(f)
                        except Exception as e:
                            reading[features_list[index_f]] = f

                    reading["activity"] = self.get_activity_label(row)
                    reading["person"] = int(self.get_name_person(row))
                    self.readings.append(reading)
        print(self.readings[0])


    def convert_csv_to_sql(self,filename, dataset_name):
        #uma_dataset_object = UmaAdlConverter()
        print("Umafall object load.")
        dataset = sqlite3.connect(filename)
        print("Starting convert object to sqlite...")
        self.dataset_dataframe.to_sql(dataset_name, dataset, if_exists='replace', index=False)
        print("Sqlite convert finished.")
        dataset.close()
    def get_name_person(self, row):
        s = row[self.row_filename_index]
        return s.split(self.char_filename_split)[self.subject_name_index]

    def get_activity_label(self, row):
        s = row[self.row_filename_index]
        return s.split(self.char_filename_split)[self.subject_activity_index]

    def filter_readings(self, readings, key, value):
        list_out = []
        for r in readings:
            if int(r[key]) == int(value):
                list_out.append(r)

        return list_out

    def convert_dataframe(self):
        data_frame = pd.DataFrame(self.readings)
        return data_frame


    def load_from_csv(self,path):
        Converter.__init__(self, path)
        self.load_data()
        self.dataset_dataframe = self.convert_dataframe()