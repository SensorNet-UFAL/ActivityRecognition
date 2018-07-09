import os
from classes.converters.arcma import ARCMAConverter

debug = True

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
#arcma.load_data()
#arcma.save_to_sql("arcma.db","arcma")
print(arcma.get_readings_by_activity("arcma.db", 1).head())

