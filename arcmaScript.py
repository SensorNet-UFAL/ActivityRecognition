import os
from classes.converters.arcma import ARCMAConverter

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
arcma.load_data()