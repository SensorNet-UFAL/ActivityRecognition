import os
from classes.converters.arcma import ARCMAConverter
from classes.commons.dataset_functions import slice_by_window
from classes.commons.classification_functions import get_dtw_mean


debug = True

arcma = ARCMAConverter("{}\\databases\\arcma".format(os.getcwd()))
arcma.load_data()
arcma.slice_to_training_test(0.8, 100)
arcma.save_to_sql("arcma.db", "arcma")
#activity_raw = arcma.get_readings_by_activity("arcma.db", 1, "x")
#activity_windows = slice_by_window(activity_raw, 50)
#dtw_mean = get_dtw_mean(activity_windows, sample)
#print("Mean1 = {}".format(dtw_mean))






