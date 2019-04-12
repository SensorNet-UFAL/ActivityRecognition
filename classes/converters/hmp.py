from classes.converters.converter import Converter
import os
import datetime
import sqlite3
import pandas as pd
from classes.commons.utils import print_debug

class HMPConverter(Converter):

    #Load TXT Data
    def load_data(self):
        folders = iter(os.walk("..\\..\\databases\\HMP_Dataset"))
        next(folders) # Skip first element with root files
        for root, dirs, files in folders:
            for file in files:
                try:
                    full_path = "{}\\{}".format(root, file)
                    fp = open(full_path, 'r')

                    line = fp.readline()
                    while line:
                        string_line = line.strip()
                        split_line = string_line.split(" ")
                        data_line = file.split("-")
                        datatime_string = data_line[1]+"/"+data_line[2]+"/"+data_line[3]+" "+data_line[4]+":"+data_line[5]+":"+data_line[6]
                        datatime_obj = datetime.datetime.strptime(datatime_string, '%Y/%m/%d %H:%M:%S')
                        self.readings.append({"time": datatime_obj, "x": int(split_line[0]), "y": int(split_line[1]), "z": int(split_line[2]), "activity": data_line[7], "person": data_line[8].split(".")[0]})
                        line = fp.readline()

                    fp.close()
                except Exception as e:
                    fp.close()
                    print(str(e))
        print_debug(self.readings[0])
        print_debug("Readings length: {}".format(len(self.readings)))
        print_debug("Converting to DataFrame...")
        self.data_frame = pd.DataFrame(self.readings)
        print_debug("Connect to SQLITE...")
        dataset = sqlite3.connect("..\\hmp.db")
        print_debug("Converting DataFrame to SQL...")
        self.data_frame.to_sql("hmp", dataset, if_exists='replace', index=False)
        print_debug("Conversion completed!")

