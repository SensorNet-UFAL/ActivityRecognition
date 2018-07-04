from classes.converters.converter import Converter

class ARCMAConverter(Converter):

    def load_data(self):
        self.load_csv_files(self, 0, ",")
