from classes.converters.converter import Converter
import os
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
                        data_line = file.split("-")
                        print("Activity: {}".format(data_line[7]))
                        break
                        line = fp.readline()

                    fp.close()
                except Exception as e:
                    fp.close()
                    print(str(e))




hmp = HMPConverter()
hmp.load_data()