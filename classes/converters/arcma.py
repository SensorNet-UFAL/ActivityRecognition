from classes.converters.converter import Converter

class ARCMAConverter(Converter):

    def load_data(self):
        #Carregando os arquivos
        self.load_csv_files(0, ",")

        #Percorrendo os arquivos carregados e armazenando as linhas
        time = 0
        for index_csv, csv_file in enumerate(self.csv_files):
            for index_row, row in enumerate(csv_file):
                self.readings.append({"time": time, "x": row[1], "y": row[2], "z": row[3], "activity": row[4]})
                time = time + 1

        print(self.readings[0])
        print("Readings length: {}".format(len(self.readings)))




